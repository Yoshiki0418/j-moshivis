"""Canonical JSONL schema used by image-understanding evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "jmoshivis.image-eval.v1"
VALID_DOMAINS = {"japan", "general_world", "academic", "other"}
VALID_TASKS = {"vqa", "multiple_choice", "caption"}


class ManifestError(ValueError):
    """Raised when a benchmark manifest is malformed."""


@dataclass(slots=True)
class EvalExample:
    """One image/question pair.

    Paths are stored relative to the manifest whenever possible.  ``source``
    preserves benchmark-specific IDs and licence metadata; ``metadata`` is for
    derived annotations such as question category or translation status.
    """

    sample_id: str
    benchmark: str
    split: str
    domain: str
    task: str
    image_path: str
    question: str
    answers: list[str]
    prompt_text: str | None = None
    audio_path: str | None = None
    choices: list[str] | None = None
    choice_labels: list[str] | None = None
    correct_choice: str | None = None
    question_type: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalExample":
        known = {
            "sample_id", "benchmark", "split", "domain", "task",
            "image_path", "question", "answers", "prompt_text", "audio_path", "choices",
            "choice_labels", "correct_choice", "question_type", "source", "metadata",
            "schema_version",
        }
        extra = sorted(set(value) - known)
        if extra:
            raise ManifestError(f"Unknown manifest fields: {', '.join(extra)}")
        try:
            example = cls(**value)
        except TypeError as exc:
            raise ManifestError(str(exc)) from exc
        example.validate()
        return example

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Keep JSONL readable and avoid treating absent optional data as data.
        for key in (
            "prompt_text", "audio_path", "choices", "choice_labels",
            "correct_choice", "question_type",
        ):
            if value[key] is None:
                del value[key]
        return value

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"{self.sample_id or '<unknown>'}: unsupported schema_version "
                f"{self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        required_strings = {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "split": self.split,
            "domain": self.domain,
            "task": self.task,
            "image_path": self.image_path,
            "question": self.question,
        }
        empty = [name for name, value in required_strings.items() if not isinstance(value, str) or not value.strip()]
        if empty:
            raise ManifestError(f"{self.sample_id or '<unknown>'}: empty fields: {', '.join(empty)}")
        if self.domain not in VALID_DOMAINS:
            raise ManifestError(f"{self.sample_id}: unknown domain {self.domain!r}")
        if self.task not in VALID_TASKS:
            raise ManifestError(f"{self.sample_id}: unknown task {self.task!r}")
        if not isinstance(self.answers, list) or not all(
            isinstance(answer, str) and answer.strip() for answer in self.answers
        ):
            raise ManifestError(f"{self.sample_id}: answers must be a non-empty list of strings")
        if self.task != "caption" and not self.answers:
            raise ManifestError(f"{self.sample_id}: {self.task} requires at least one answer")
        if self.task == "multiple_choice":
            if not self.choices or self.correct_choice is None:
                raise ManifestError(
                    f"{self.sample_id}: multiple_choice requires choices and correct_choice"
                )
            labels = self.choice_labels or _choice_labels(len(self.choices))
            if len(labels) != len(self.choices) or len(set(labels)) != len(labels):
                raise ManifestError(
                    f"{self.sample_id}: choice_labels must be unique and match choices"
                )
            if self.correct_choice not in self.choices and self.correct_choice not in labels:
                raise ManifestError(
                    f"{self.sample_id}: correct_choice must be a choice value or label"
                )
        if not isinstance(self.source, dict) or not isinstance(self.metadata, dict):
            raise ManifestError(f"{self.sample_id}: source and metadata must be objects")
        if self.prompt_text is not None and not self.prompt_text.strip():
            raise ManifestError(f"{self.sample_id}: prompt_text must not be empty")

    @property
    def rendered_prompt(self) -> str:
        """The one prompt string shared by text input and TTS input."""

        return self.prompt_text if self.prompt_text is not None else self.question

    def resolve_image(self, manifest_path: str | Path) -> Path:
        return resolve_path(self.image_path, manifest_path)

    def resolve_audio(self, manifest_path: str | Path) -> Path | None:
        return None if self.audio_path is None else resolve_path(self.audio_path, manifest_path)


def _choice_labels(size: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(size)]


def resolve_path(value: str, manifest_path: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def read_manifest(
    path: str | Path,
    *,
    check_files: bool = False,
    require_audio: bool = False,
) -> list[EvalExample]:
    examples: list[EvalExample] = []
    seen: set[str] = set()
    for line_number, value in read_jsonl(path):
        try:
            example = EvalExample.from_dict(value)
        except ManifestError as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
        if example.sample_id in seen:
            raise ManifestError(f"{path}:{line_number}: duplicate sample_id {example.sample_id!r}")
        seen.add(example.sample_id)
        if require_audio and not example.audio_path:
            raise ManifestError(f"{path}:{line_number}: {example.sample_id} has no audio_path")
        if check_files:
            image = example.resolve_image(path)
            if not image.is_file():
                raise ManifestError(f"{path}:{line_number}: image not found: {image}")
            audio = example.resolve_audio(path)
            if require_audio and (audio is None or not audio.is_file()):
                raise ManifestError(f"{path}:{line_number}: audio not found: {audio}")
        examples.append(example)
    if not examples:
        raise ManifestError(f"{path}: manifest is empty")
    return examples


def write_manifest(path: str | Path, examples: Iterable[EvalExample]) -> tuple[int, str]:
    """Write a manifest atomically and return ``(record_count, sha256)``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    seen: set[str] = set()
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            example.validate()
            if example.sample_id in seen:
                raise ManifestError(f"duplicate sample_id {example.sample_id!r}")
            seen.add(example.sample_id)
            encoded = json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
            count += 1
        handle.flush()
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ManifestError("refusing to write an empty manifest")
    temporary.replace(destination)
    return count, digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_question_audio_provenance(example: EvalExample) -> None:
    """Verify that question-audio metadata names the prompt it synthesized.

    ``prepare_tts`` records the synthesized prompt as ``prompt_text`` on both
    the example and ``metadata.question_audio``.  In particular, a
    ``short_answer`` prompt is intentionally different from ``question`` and
    must be checked against ``rendered_prompt``, not the original question.

    Existing manifests call the text digest ``question_sha256``.  Also accept
    the less ambiguous ``prompt_sha256`` spelling, while checking every digest
    that is present so a stale compatibility field cannot be hidden.
    """

    audio = example.metadata.get("question_audio")
    if not isinstance(audio, dict):
        raise ManifestError(
            f"{example.sample_id}: metadata.question_audio must be an object"
        )

    recorded_sample_id = audio.get("sample_id")
    if recorded_sample_id != example.sample_id:
        raise ManifestError(
            f"{example.sample_id}: metadata.question_audio.sample_id mismatch "
            f"({recorded_sample_id!r})"
        )

    rendered_prompt = example.rendered_prompt
    recorded_prompt = audio.get("prompt_text")
    if recorded_prompt != rendered_prompt:
        raise ManifestError(
            f"{example.sample_id}: metadata.question_audio.prompt_text does not "
            "match the rendered prompt"
        )

    expected_digest = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    digest_fields = [
        field for field in ("question_sha256", "prompt_sha256") if field in audio
    ]
    if not digest_fields:
        raise ManifestError(
            f"{example.sample_id}: metadata.question_audio.question_sha256 "
            "or prompt_sha256 is missing"
        )
    for field in digest_fields:
        if audio[field] != expected_digest:
            raise ManifestError(
                f"{example.sample_id}: metadata.question_audio.{field} does not "
                "match the rendered prompt"
            )
