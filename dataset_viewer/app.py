"""aiohttp application for the read-only J-MoshiVis dataset viewer.

This module deliberately imports neither the training stack nor GPU libraries.  The
catalog is the only component allowed to turn an opaque item token into a media
path; request parameters are never interpreted as filesystem paths.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import mimetypes
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aiohttp import web

if __package__ in {None, ""}:  # pragma: no cover - direct script entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset_viewer.catalog import DatasetCatalog
    from ssvd.review_store import MAX_REVIEW_NOTE_CHARS, ReviewConflictError
else:
    from .catalog import DatasetCatalog
    from ssvd.review_store import MAX_REVIEW_NOTE_CHARS, ReviewConflictError


LOGGER = logging.getLogger("dataset_viewer")


@dataclass
class _RefreshState:
    lock: asyncio.Lock
    task: asyncio.Task[None] | None = None
    periodic_task: asyncio.Task[None] | None = None
    error: str | None = None
    last_interaction: float = float("-inf")


APP_CATALOG: web.AppKey[Any] = web.AppKey("catalog")
APP_REPO_ROOT: web.AppKey[Path] = web.AppKey("repo_root")
APP_REFRESH_STATE: web.AppKey[_RefreshState] = web.AppKey("refresh_state")
APP_REBUILD: web.AppKey[bool] = web.AppKey("rebuild")
APP_REFRESH_INTERVAL: web.AppKey[float] = web.AppKey("refresh_interval")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_LIMIT = 40
MAX_LIMIT = 100
MAX_OFFSET = 10_000_000
DEFAULT_REFRESH_INTERVAL = 10.0

_SAFE_MEDIA_KIND = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_IMAGE_MIMES = frozenset(
    {
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
_AUDIO_MIMES = frozenset(
    {
        "audio/flac",
        "audio/m4a",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "audio/x-flac",
        "audio/x-m4a",
        "audio/x-wav",
    }
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_response(payload: Any, *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, dumps=_json_dumps)


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "message": message, "error": {"code": code, "message": message}}


@web.middleware
async def _error_middleware(
    request: web.Request, handler: Callable[[web.Request], Any]
) -> web.StreamResponse:
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        if request.path.startswith("/api/"):
            response = _json_response(
                _error_payload(exc.reason.lower().replace(" ", "_"), exc.reason),
                status=exc.status,
            )
        else:
            raise
    except ValueError as exc:
        response = _json_response(_error_payload("bad_request", str(exc)), status=400)
    except (KeyError, FileNotFoundError):
        response = _json_response(
            _error_payload("not_found", "The requested dataset item was not found."),
            status=404,
        )
    except Exception:
        LOGGER.exception("Unhandled request error: %s %s", request.method, request.path)
        response = _json_response(
            _error_payload("internal_error", "The request could not be completed."),
            status=500,
        )

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    if request.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _method(catalog: Any, *names: str) -> Callable[..., Any]:
    for name in names:
        value = getattr(catalog, name, None)
        if callable(value):
            return value
    raise RuntimeError(f"Catalog does not implement any of: {', '.join(names)}")


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a catalog method while tolerating older catalogs with fewer filters."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**supported)


def _catalog_status(catalog: Any) -> dict[str, Any]:
    result = _method(catalog, "status", "get_status")()
    if not isinstance(result, Mapping):
        raise RuntimeError("Catalog status must be a mapping")
    return dict(result)


def _parse_int(
    raw: str | None,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return min(value, maximum)


def _optional_bool(raw: str | None, *, name: str) -> bool | None:
    if raw is None or raw == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


async def _run_refresh(app: web.Application, *, force: bool) -> None:
    state = app[APP_REFRESH_STATE]
    async with state.lock:
        state.error = None
        try:
            refresh = _method(app[APP_CATALOG], "refresh")
            await asyncio.to_thread(_call_with_supported_kwargs, refresh, force=force)
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Dataset catalog refresh failed")


def _start_refresh(app: web.Application, *, force: bool) -> bool:
    state = app[APP_REFRESH_STATE]
    current = state.task
    if current is not None and not current.done():
        return False

    task = asyncio.create_task(_run_refresh(app, force=force), name="dataset-catalog-refresh")
    state.task = task

    def _clear(completed: asyncio.Task[None]) -> None:
        # Retrieve the exception to prevent an unobserved-task warning.  Errors in
        # the refresh body are already converted into status information.
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(_clear)
    return True


async def _on_startup(app: web.Application) -> None:
    _start_refresh(app, force=app[APP_REBUILD])
    interval = app[APP_REFRESH_INTERVAL]
    if interval > 0:
        app[APP_REFRESH_STATE].periodic_task = asyncio.create_task(
            _periodic_refresh(app, interval),
            name="dataset-catalog-periodic-refresh",
        )


async def _periodic_refresh(app: web.Application, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        # Indexing filesystem trees must not contend with a reviewer saving
        # every few seconds. Manual refresh remains available at any time.
        if asyncio.get_running_loop().time() - app[APP_REFRESH_STATE].last_interaction < 30:
            continue
        # _start_refresh returns False while another refresh is running, so a
        # slow scan never overlaps the next periodic tick.
        _start_refresh(app, force=False)


async def _on_cleanup(app: web.Application) -> None:
    state = app[APP_REFRESH_STATE]
    periodic_task = state.periodic_task
    if periodic_task is not None and not periodic_task.done():
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass

    task = state.task
    if task is not None and not task.done():
        # asyncio.to_thread cannot stop a running scanner.  We do not wait for it
        # here; cancelling only detaches the request-side task during shutdown.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    close = getattr(app[APP_CATALOG], "close", None)
    if callable(close):
        close()


async def health(_: web.Request) -> web.Response:
    return _json_response({"ok": True, "service": "j-moshivis-dataset-viewer"})


async def status(request: web.Request) -> web.Response:
    payload = await asyncio.to_thread(_catalog_status, request.app[APP_CATALOG])
    state = request.app[APP_REFRESH_STATE]
    task = state.task
    payload["refresh"] = {
        "running": task is not None and not task.done(),
        "last_error": state.error,
        "interval_seconds": request.app[APP_REFRESH_INTERVAL],
    }
    return _json_response(payload)


async def dashboard(request: web.Request) -> web.Response:
    catalog = request.app[APP_CATALOG]
    payload = await asyncio.to_thread(_call_with_supported_kwargs,
        _method(catalog, "dashboard", "get_dashboard"),
        collection=request.rel_url.query.get("collection"),
        category=request.rel_url.query.get("category"),
    )
    return _json_response(payload)


async def tree(request: web.Request) -> web.Response:
    query = request.rel_url.query
    catalog = request.app[APP_CATALOG]
    payload = await asyncio.to_thread(_call_with_supported_kwargs,
        _method(catalog, "tree", "get_tree"),
        collection=query.get("collection"),
        parent=query.get("parent"),
        category=query.get("category"),
    )
    return _json_response(payload)


def _item_filters(request: web.Request) -> dict[str, Any]:
    query = request.rel_url.query
    filters: dict[str, Any] = {
        "collection": query.get("collection"),
        "category": query.get("category"),
        "directory": query.get("directory") or query.get("path"),
        "stage": query.get("stage"),
        "query": query.get("q") or query.get("query"),
        "has_image": _optional_bool(query.get("has_image"), name="has_image"),
        "has_dialogue": _optional_bool(query.get("has_dialogue"), name="has_dialogue"),
        "has_audio": _optional_bool(query.get("has_audio"), name="has_audio"),
        "has_alignment": _optional_bool(
            query.get("has_alignment"), name="has_alignment"
        ),
        "review_status": query.get("review_status"),
        "review_required": _optional_bool(
            query.get("review_required"), name="review_required"
        ),
        "auxiliary_kind": query.get("auxiliary_kind"),
    }
    return {key: value for key, value in filters.items() if value is not None}


async def items(request: web.Request) -> web.Response:
    limit = _parse_int(
        request.rel_url.query.get("limit"),
        name="limit",
        default=DEFAULT_LIMIT,
        minimum=1,
        maximum=MAX_LIMIT,
    )
    page = _parse_int(
        request.rel_url.query.get("page"),
        name="page",
        default=1,
        minimum=1,
        maximum=MAX_OFFSET // max(1, limit) + 1,
    )
    offset = _parse_int(
        request.rel_url.query.get("offset"),
        name="offset",
        default=(page - 1) * limit,
        minimum=0,
        maximum=MAX_OFFSET,
    )
    if request.rel_url.query.get("page") in {None, ""} and offset:
        page = offset // limit + 1
    filters = _item_filters(request)
    payload = await asyncio.to_thread(_call_with_supported_kwargs,
        _method(request.app[APP_CATALOG], "list_items", "items"),
        limit=limit,
        page=page,
        offset=offset,
        filters=filters,
        **filters,
    )
    return _json_response(payload)


async def item_detail(request: web.Request) -> web.Response:
    request.app[APP_REFRESH_STATE].last_interaction = asyncio.get_running_loop().time()
    token = request.match_info["token"]
    if not token:
        raise ValueError("token is required")
    filters = _item_filters(request)
    payload = await asyncio.to_thread(_call_with_supported_kwargs,
        _method(request.app[APP_CATALOG], "get_item", "item"),
        token=token,
        filters=filters,
        **filters,
    )
    if payload is None:
        raise KeyError(token)
    return _json_response(payload)


async def review_item(request: web.Request) -> web.Response:
    request.app[APP_REFRESH_STATE].last_interaction = asyncio.get_running_loop().time()
    token = request.match_info["token"]
    if not token:
        raise ValueError("token is required")
    if request.content_type != "application/json":
        raise ValueError("Content-Type must be application/json")
    body = await request.json()
    if not isinstance(body, Mapping):
        raise ValueError("JSON body must be an object")
    content_sha256 = body.get("content_sha256")
    fingerprint_alias = body.get("fingerprint")
    if (
        content_sha256 is not None
        and fingerprint_alias is not None
        and content_sha256 != fingerprint_alias
    ):
        raise ValueError("content_sha256 and fingerprint must match")
    fingerprint = content_sha256 or fingerprint_alias
    decision = body.get("decision")
    note = body.get("note", "")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("content_sha256 is required")
    if not isinstance(decision, str) or not decision:
        raise ValueError("decision is required")
    if not isinstance(note, str):
        raise ValueError("note must be a string")
    if len(note.strip()) > MAX_REVIEW_NOTE_CHARS:
        raise ValueError(
            f"note must be at most {MAX_REVIEW_NOTE_CHARS} characters"
        )
    reviewer = body.get("reviewer")
    if reviewer is not None and not isinstance(reviewer, str):
        raise ValueError("reviewer must be a string")
    try:
        payload = await asyncio.to_thread(
            _call_with_supported_kwargs,
            _method(request.app[APP_CATALOG], "review_item"),
            token=token,
            fingerprint=fingerprint,
            decision=decision,
            note=note,
            reviewer=reviewer or request.remote,
        )
    except ReviewConflictError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    return _json_response(payload)


async def edit_dialogue(request: web.Request) -> web.Response:
    request.app[APP_REFRESH_STATE].last_interaction = asyncio.get_running_loop().time()
    if request.content_type != "application/json":
        raise ValueError("Content-Type must be application/json")
    body = await request.json()
    if not isinstance(body, Mapping):
        raise ValueError("JSON body must be an object")
    fingerprint = body.get("content_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("content_sha256 is required")
    if not isinstance(body.get("dialogue"), list):
        raise ValueError("dialogue must be an array")
    if body.get("reviewer") is not None and not isinstance(body["reviewer"], str):
        raise ValueError("reviewer must be a string")
    try:
        payload = await asyncio.to_thread(
            request.app[APP_CATALOG].edit_dialogue,
            request.match_info["token"], fingerprint=fingerprint,
            dialogue=body["dialogue"], auxiliary_task=body.get("auxiliary_task"),
            note=body.get("note", ""), reviewer=body.get("reviewer") or request.remote,
        )
    except ReviewConflictError as exc:
        raise web.HTTPConflict(reason=str(exc)) from exc
    return _json_response(payload)


def _resolved_media_path(result: Any) -> tuple[Path, str | None]:
    supplied_mime: str | None = None
    if isinstance(result, Mapping):
        raw_path = result.get("path") or result.get("file")
        raw_mime = result.get("mime_type") or result.get("content_type")
        if isinstance(raw_mime, str):
            supplied_mime = raw_mime.lower().split(";", 1)[0].strip()
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        raw_path = result[0] if result else None
        if len(result) > 1 and isinstance(result[1], str):
            supplied_mime = result[1].lower().split(";", 1)[0].strip()
    else:
        raw_path = result
    if not isinstance(raw_path, (str, Path)):
        raise FileNotFoundError("Catalog returned no media path")
    return Path(raw_path), supplied_mime


async def media(request: web.Request) -> web.StreamResponse:
    token = request.match_info["token"]
    kind = request.match_info["kind"]
    if not _SAFE_MEDIA_KIND.fullmatch(kind):
        raise ValueError("Unsupported media kind")

    result = _call_with_supported_kwargs(
        _method(request.app[APP_CATALOG], "resolve_media", "get_media"),
        token=token,
        kind=kind,
    )
    if result is None:
        raise KeyError((token, kind))
    path, supplied_mime = _resolved_media_path(result)
    if not path.is_absolute():
        path = request.app[APP_REPO_ROOT] / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    guessed_mime = mimetypes.guess_type(path.name)[0]
    mime = supplied_mime or guessed_mime
    if mime is None:
        raise ValueError("The registered media file has an unknown type")
    mime = mime.lower().split(";", 1)[0].strip()
    allowed = _IMAGE_MIMES | _AUDIO_MIMES
    if mime not in allowed:
        raise ValueError("The registered file is not a supported image or audio type")

    response = web.FileResponse(
        path,
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{path.name.replace(chr(34), "")}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
    response.content_type = mime
    return response


async def refresh(request: web.Request) -> web.Response:
    force = _optional_bool(request.rel_url.query.get("force"), name="force") or False
    if request.can_read_body and request.content_type == "application/json":
        body = await request.json()
        if not isinstance(body, Mapping):
            raise ValueError("JSON body must be an object")
        if "force" in body:
            body_force = body["force"]
            if not isinstance(body_force, bool):
                raise ValueError("force must be a boolean")
            force = body_force
    started = _start_refresh(request.app, force=force)
    return _json_response(
        {
            "ok": True,
            "accepted": started,
            "message": "refresh started" if started else "refresh already running",
        },
        status=202,
    )


def _static_response(static_dir: Path, filename: str) -> web.FileResponse:
    # Only callers with hard-coded file names reach this helper.  Resolving and
    # checking the parent retains StaticResource's follow_symlinks=False property.
    root = static_dir.resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(
        path,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )


def _static_handler(static_dir: Path, filename: str) -> Callable[[web.Request], Any]:
    async def handler(_: web.Request) -> web.FileResponse:
        return _static_response(static_dir, filename)

    return handler


def create_app(
    *,
    repo_root: str | Path | None = None,
    db_path: str | Path | None = None,
    catalog: Any | None = None,
    rebuild: bool = False,
    start_background_refresh: bool = True,
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
) -> web.Application:
    """Build the web application.

    ``catalog`` and ``start_background_refresh`` are dependency-injection hooks
    used by the API tests; normal callers should leave both at their defaults.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    database = Path(db_path or root / "data" / ".dataset_viewer" / "catalog.sqlite3")
    if not database.is_absolute():
        database = root / database
    if catalog is None:
        catalog = DatasetCatalog(repo_root=root, db_path=database)

    app = web.Application(middlewares=[_error_middleware])
    app[APP_CATALOG] = catalog
    app[APP_REPO_ROOT] = root
    app[APP_REFRESH_STATE] = _RefreshState(lock=asyncio.Lock())
    app[APP_REBUILD] = rebuild
    app[APP_REFRESH_INTERVAL] = max(0.0, float(refresh_interval))

    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/dashboard", dashboard)
    app.router.add_get("/api/tree", tree)
    app.router.add_get("/api/items", items)
    app.router.add_get("/api/items/{token}", item_detail)
    app.router.add_put("/api/items/{token}/review", review_item)
    app.router.add_put("/api/items/{token}/dialogue", edit_dialogue)
    app.router.add_get("/api/items/{token}/media/{kind}", media)
    app.router.add_post("/api/refresh", refresh)

    static_dir = Path(__file__).with_name("static")
    app.router.add_get("/", _static_handler(static_dir, "index.html"))
    app.router.add_get("/static/styles.css", _static_handler(static_dir, "styles.css"))
    app.router.add_get("/static/app.js", _static_handler(static_dir, "app.js"))

    if start_background_refresh:
        app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse J-MoshiVis dataset construction progress (read-only)."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="J-MoshiVis repository root",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="catalog SQLite path (default: <repo>/data/.dataset_viewer/catalog.sqlite3)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force a full catalog rebuild in the startup background refresh",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=DEFAULT_REFRESH_INTERVAL,
        help=(
            "seconds between lightweight background refreshes "
            f"(default: {DEFAULT_REFRESH_INTERVAL:g}; 0 disables)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.refresh_interval < 0:
        raise SystemExit("--refresh-interval must be zero or greater")
    web.run_app(
        create_app(
            repo_root=args.repo_root,
            db_path=args.db,
            rebuild=args.rebuild,
            refresh_interval=args.refresh_interval,
        ),
        host=args.host,
        port=args.port,
        print=lambda message: LOGGER.info("%s", message),
    )


if __name__ == "__main__":
    main()
