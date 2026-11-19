"use strict";

const PAGE_SIZE = 50;
const STATUS_POLL_INTERVAL_MS = 10000;
const DETAIL_PREFETCH_TTL_MS = 30000;
const DETAIL_PREFETCH_LIMIT = 3;
const STAGES = [
  { key: "image", label: "画像" },
  { key: "dialogue", label: "対話" },
  { key: "audio", label: "音声" },
  { key: "alignment", label: "アライメント" },
];
const COLLECTION_ORDER = ["visual-v3", "visual_v3", "japanese-daily-v1", "japanese_daily_v1", "counting-v4", "counting_v4", "ocr-v4", "ocr_v4", "nonvisual-v1", "nonvisual_v1"];
const COLLECTION_LABELS = {
  "visual-v3": "visual-v3",
  visual_v3: "visual-v3",
  "japanese-daily-v1": "日本の日常画像-v1",
  japanese_daily_v1: "日本の日常画像-v1",
  "counting-v4": "counting-v4",
  counting_v4: "counting-v4",
  "ocr-v4": "OCR-v4",
  ocr_v4: "OCR-v4",
  "nonvisual-v1": "nonvisual-v1",
  nonvisual_v1: "nonvisual-v1",
};

const state = {
  page: 1,
  pages: 1,
  total: 0,
  limit: PAGE_SIZE,
  items: [],
  selectedToken: "",
  selectedItem: null,
  editingDialogue: false,
  savingDialogue: false,
  reviewBusy: false,
  annotationMode: false,
  annotationListHidden: false,
  annotationAdvanced: false,
  filters: {
    collection: "", category: "", directory: "", stage: "", review_status: "",
    auxiliary_kind: "", review_required: "", ai_status: "", q: "",
  },
  navigation: { previous: "", next: "" },
  listController: null,
  detailController: null,
  detailRequestId: 0,
  listRequestId: 0,
  detailPrefetch: new Map(),
  excludedReviewTokens: new Set(),
  reviewQueueScope: "",
  statusWasBusy: false,
  manualRefreshPending: false,
  catalogRevision: null,
  aiSuggestionCount: null,
  aiListRefreshPending: false,
  statusPollTimer: null,
  statusPollRunning: false,
  collectionStats: new Map(),
  toastTimer: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  collectDom();
  bindEvents();
  setViewMode(window.location.hash === "#annotate", { reload: false, updateUrl: false });
  void initialize();
});

function collectDom() {
  const ids = [
    "index-state", "index-state-label", "refresh-button", "last-updated",
    "auto-refresh-note", "auto-refresh-label", "metric-total-audio",
    "metric-total-audio-duration", "metric-total-audio-sub",
    "metric-visual-images", "metric-visual-images-sub",
    "metric-visual-dialogues", "metric-visual-dialogues-sub",
    "metric-visual-audio", "metric-visual-audio-sub",
    "metric-visual-alignments", "metric-visual-alignments-sub",
    "metric-nonvisual-dialogues", "metric-nonvisual-dialogues-sub",
    "metric-nonvisual-audio", "metric-nonvisual-audio-sub",
    "metric-nonvisual-alignments", "metric-nonvisual-alignments-sub",
    "metric-review-required", "metric-review-pending", "metric-review-certified", "metric-review-certified-sub",
    "metric-review-regeneration", "metric-review-regeneration-sub", "metric-review-optional", "metric-review-rejected",
    "filter-form", "category-filter", "stage-filter", "uid-search", "clear-filters",
    "review-status-filter", "auxiliary-kind-filter", "review-required-filter", "ai-status-filter",
    "collection-tree", "all-collections", "all-count", "active-collection", "result-count",
    "list-loading", "list-empty", "list-error", "list-error-message", "list-retry",
    "dataset-list", "page-prev", "page-next", "page-current", "page-total",
    "tree-toggle", "tree-close", "detail-panel", "detail-empty", "detail-content",
    "detail-close", "detail-loading", "detail-category", "detail-uid", "detail-path",
    "item-prev", "item-next", "detail-stepper", "detail-image", "image-missing",
    "image-availability", "detail-dialogue", "dialogue-availability", "detail-audio",
    "audio-card", "audio-missing", "audio-availability", "alignment-wrap",
    "alignment-body", "alignment-missing", "alignment-availability", "metadata-grid",
    "review-section", "review-status-badge", "review-description", "review-facts", "review-qa",
    "review-usage", "review-note", "review-error", "review-actions",
    "review-approve", "review-reject", "review-reset", "review-edit",
    "dialogue-editor-actions", "dialogue-editor-facts", "dialogue-save", "dialogue-cancel", "dialogue-editor-error",
    "mobile-backdrop", "toast",
    "browse-mode", "annotation-mode", "annotation-collection", "annotation-auto-next", "annotation-list-toggle",
    "annotation-image-open",
    "annotation-advanced-toggle",
    "dialogue-heading-actions", "review-context-details",
    "audio-section", "alignment-section",
    "ai-suggestion", "ai-suggestion-label", "ai-suggestion-concerns", "ai-suggestion-diff",
  ];
  for (const id of ids) dom[toCamel(id)] = document.getElementById(id);
  dom.treePanel = document.querySelector(".tree-panel");
}

function bindEvents() {
  dom.aiStatusFilter.addEventListener("change", () => {
    state.filters.ai_status = dom.aiStatusFilter.value;
    if (state.filters.ai_status) {
      state.filters.review_status = "pending";
      state.filters.review_required = "true";
      dom.reviewStatusFilter.value = "pending";
      dom.reviewRequiredFilter.checked = true;
    }
    state.page = 1;
    clearDetailPrefetch();
    void loadItems({ selectFirst: true });
  });
  dom.browseMode.addEventListener("click", () => setViewMode(false));
  dom.annotationMode.addEventListener("click", () => setViewMode(true));
  window.addEventListener("hashchange", () => setViewMode(window.location.hash === "#annotate", { updateUrl: false }));
  dom.annotationListToggle.addEventListener("click", () => {
    state.annotationListHidden = !state.annotationListHidden;
    updateAnnotationLayout();
  });
  dom.annotationAdvancedToggle.addEventListener("click", () => {
    state.annotationAdvanced = !state.annotationAdvanced;
    updateAnnotationLayout();
  });
  dom.annotationCollection.addEventListener("change", () => {
    selectTreeFilters({ collection: dom.annotationCollection.value }, dom.annotationCollection.selectedOptions[0]?.textContent);
  });
  dom.filterForm.addEventListener("submit", (event) => {
    event.preventDefault();
    state.filters.category = dom.categoryFilter.value;
    state.filters.stage = dom.stageFilter.value;
    state.filters.review_status = dom.reviewStatusFilter.value;
    state.filters.auxiliary_kind = dom.auxiliaryKindFilter.value;
    state.filters.review_required = dom.reviewRequiredFilter.checked ? "true" : "";
    if (dom.reviewRequiredFilter.checked) {
      state.filters.review_status = "pending";
      dom.reviewStatusFilter.value = "pending";
    }
    state.filters.q = dom.uidSearch.value.trim();
    state.page = 1;
    void loadItems({ selectFirst: true });
  });

  dom.categoryFilter.addEventListener("change", () => {
    state.filters.category = dom.categoryFilter.value;
    state.filters.directory = "";
    state.page = 1;
    updateTreeSelection();
    void loadItems({ selectFirst: true });
  });
  dom.stageFilter.addEventListener("change", () => {
    state.filters.stage = dom.stageFilter.value;
    state.page = 1;
    void loadItems({ selectFirst: true });
  });
  dom.reviewStatusFilter.addEventListener("change", () => {
    if (dom.reviewStatusFilter.value !== "pending") {
      dom.reviewRequiredFilter.checked = false;
      state.filters.review_required = "";
      state.filters.ai_status = "";
      dom.aiStatusFilter.value = "";
    }
    state.filters.review_status = dom.reviewStatusFilter.value;
    state.page = 1;
    void loadItems({ selectFirst: true });
  });
  dom.auxiliaryKindFilter.addEventListener("change", () => {
    state.filters.auxiliary_kind = dom.auxiliaryKindFilter.value;
    state.page = 1;
    void loadItems({ selectFirst: true });
  });
  dom.reviewRequiredFilter.addEventListener("change", () => {
    state.filters.review_required = dom.reviewRequiredFilter.checked ? "true" : "";
    if (!dom.reviewRequiredFilter.checked) {
      state.filters.ai_status = "";
      dom.aiStatusFilter.value = "";
    }
    if (dom.reviewRequiredFilter.checked) {
      state.filters.review_status = "pending";
      dom.reviewStatusFilter.value = "pending";
    }
    state.page = 1;
    void loadItems({ selectFirst: true });
  });
  dom.clearFilters.addEventListener("click", clearFilters);
  dom.refreshButton.addEventListener("click", () => void refreshIndex());
  dom.listRetry.addEventListener("click", () => void loadItems());

  dom.pagePrev.addEventListener("click", () => {
    if (state.page > 1) {
      state.page -= 1;
      void loadItems({ scrollTop: true, selectFirst: true });
    }
  });
  dom.pageNext.addEventListener("click", () => {
    if (state.page < state.pages) {
      state.page += 1;
      void loadItems({ scrollTop: true, selectFirst: true });
    }
  });

  dom.allCollections.addEventListener("click", () => selectTreeFilters({}, "ALL DATA"));
  dom.itemPrev.addEventListener("click", () => navigateDetail("previous"));
  dom.itemNext.addEventListener("click", () => navigateDetail("next"));
  dom.detailClose.addEventListener("click", closeDetail);
  dom.treeToggle.addEventListener("click", openTree);
  dom.treeClose.addEventListener("click", closeTree);
  dom.mobileBackdrop.addEventListener("click", closeOverlays);

  dom.detailAudio.addEventListener("timeupdate", highlightCurrentAlignment);
  dom.detailAudio.addEventListener("ended", clearActiveAlignment);
  dom.audioSection.addEventListener("toggle", () => {
    if (!dom.audioSection.open) dom.detailAudio.pause();
  });
  dom.reviewApprove.addEventListener("click", () => void submitReview("approved"));
  dom.reviewReject.addEventListener("click", () => void submitReview("rejected"));
  dom.reviewReset.addEventListener("click", () => void submitReview("pending"));
  dom.reviewEdit.addEventListener("click", startDialogueEdit);
  dom.dialogueSave.addEventListener("click", () => void saveDialogueEdit());
  dom.dialogueCancel.addEventListener("click", cancelDialogueEdit);
  document.querySelectorAll("[data-review-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      clearFilters(false);
      const filter = button.dataset.reviewFilter;
      state.filters.review_status = filter === "regeneration" ? "approved" : filter;
      state.filters.stage = filter === "regeneration" ? "regeneration_pending" : "";
      state.filters.review_required = filter === "pending" ? "true" : "";
      dom.reviewStatusFilter.value = state.filters.review_status;
      dom.stageFilter.value = state.filters.stage;
      dom.reviewRequiredFilter.checked = filter === "pending";
      void loadItems({ selectFirst: true });
    });
  });
  window.addEventListener("beforeunload", (event) => {
    if (state.editingDialogue) { event.preventDefault(); event.returnValue = ""; }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeOverlays();
      return;
    }
    const target = event.target;
    const editing = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement || target instanceof HTMLAudioElement;
    if (editing || !state.selectedToken || !event.altKey) return;
    if (event.key === "ArrowLeft" && state.navigation.previous) {
      event.preventDefault();
      navigateDetail("previous");
    }
    if (event.key === "ArrowRight" && state.navigation.next) {
      event.preventDefault();
      navigateDetail("next");
    }
  });

  window.addEventListener("resize", () => { updateBackdrop(); scheduleDialogueEditorSizing(); });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      window.clearTimeout(state.statusPollTimer);
      void pollStatus();
    }
  });
}

async function initialize() {
  const results = await Promise.allSettled([
    loadStatus(),
    loadDashboard(),
    loadTree(),
  ]);
  if (results.some((result) => result.status === "rejected")) {
    showToast("一部の集計情報を取得できませんでした。データ一覧は引き続き利用できます。", true);
  }
  await loadItems();
  scheduleStatusPoll();
}

function scheduleStatusPoll() {
  window.clearTimeout(state.statusPollTimer);
  state.statusPollTimer = window.setTimeout(() => void pollStatus(), STATUS_POLL_INTERVAL_MS);
}

async function pollStatus() {
  if (state.statusPollRunning) return;
  if (document.hidden) {
    scheduleStatusPoll();
    return;
  }
  state.statusPollRunning = true;
  try {
    await loadStatus({ quiet: true });
  } finally {
    state.statusPollRunning = false;
    scheduleStatusPoll();
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || body.error?.message || message;
    } catch (_) {
      // The status text remains the most useful fallback.
    }
    throw new Error(message);
  }
  if (response.status === 204) return {};
  return response.json();
}

async function loadStatus({ quiet = false } = {}) {
  try {
    const raw = await api("/api/status");
    const status = raw.status || raw.state || (raw.indexing ? "indexing" : "ready");
    const busy = ["indexing", "refreshing", "busy", "running", "loading"].includes(String(status).toLowerCase()) || raw.indexing === true || raw.refresh?.running === true || raw.ready === false;
    const refreshError = raw.refresh?.last_error;
    const failed = ["error", "failed"].includes(String(status).toLowerCase()) || raw.ok === false || Boolean(refreshError);
    const nextRevision = asNumber(raw.catalog_revision ?? raw.revision);
    const revisionChanged = state.catalogRevision !== null && nextRevision !== null && nextRevision !== state.catalogRevision;
    if (nextRevision !== null) state.catalogRevision = nextRevision;
    const aiCount = asNumber(raw.ai_suggestion_count);
    if (state.aiSuggestionCount !== null && aiCount !== null && aiCount !== state.aiSuggestionCount) {
      state.aiListRefreshPending = true;
    }
    if (aiCount !== null) state.aiSuggestionCount = aiCount;
    const manualRefreshCompleted = state.manualRefreshPending && state.statusWasBusy && !busy;

    dom.indexState.classList.toggle("is-ready", !busy && !failed);
    dom.indexState.classList.toggle("is-busy", busy);
    dom.indexState.classList.toggle("is-error", failed);
    dom.indexStateLabel.textContent = busy
      ? (raw.message || "索引を更新中…")
      : failed
        ? (raw.message || refreshError || "索引エラー")
        : (raw.message || "索引は最新です");
    dom.refreshButton.disabled = busy;

    const refreshInterval = asNumber(raw.refresh?.interval_seconds);
    if (refreshInterval !== null) {
      dom.autoRefreshNote.classList.toggle("is-disabled", refreshInterval <= 0);
      dom.autoRefreshLabel.textContent = refreshInterval > 0
        ? `${formatNumber(refreshInterval)}秒ごとに自動更新`
        : "自動更新は停止中";
    }

    const updated = raw.updated_at || raw.last_updated || raw.indexed_at || raw.last_refresh;
    if (updated) dom.lastUpdated.textContent = `最終確認: ${formatDateTime(updated)}`;

    if ((revisionChanged || manualRefreshCompleted) && !failed) {
      clearDetailPrefetch();
      if (state.annotationMode && !manualRefreshCompleted) refreshAfterAnnotation();
      else await Promise.allSettled([loadDashboard(), loadTree(), loadItems()]);
      if (manualRefreshCompleted) showToast("再索引が完了しました。");
    }
    if (manualRefreshCompleted || (state.manualRefreshPending && failed)) {
      if (failed) showToast(`再索引に失敗しました: ${refreshError || "不明なエラー"}`, true);
      state.manualRefreshPending = false;
    }
    if (state.aiListRefreshPending && state.filters.ai_status && !state.editingDialogue
        && !state.savingDialogue && !state.reviewBusy && !failed) {
      state.aiListRefreshPending = false;
      clearDetailPrefetch();
      // Only refresh the queue. A newly completed AI draft must never replace
      // the current text/draft that a person is already inspecting.
      await loadItems();
    }
    state.statusWasBusy = busy;
    prefetchNextDetails();
    return raw;
  } catch (error) {
    dom.indexState.classList.remove("is-ready", "is-busy");
    dom.indexState.classList.add("is-error");
    dom.indexStateLabel.textContent = "サーバーに接続できません";
    if (!quiet) throw error;
    return null;
  }
}

async function refreshIndex() {
  dom.refreshButton.disabled = true;
  dom.indexState.classList.remove("is-ready", "is-error");
  dom.indexState.classList.add("is-busy");
  dom.indexStateLabel.textContent = "再索引を開始しています…";
  try {
    const result = await api("/api/refresh", { method: "POST" });
    const asyncRefresh = result.accepted === true || result.status === "indexing" || result.indexing === true;
    const alreadyRunning = result.accepted === false;
    state.statusWasBusy = asyncRefresh || alreadyRunning;
    state.manualRefreshPending = asyncRefresh || alreadyRunning;
    if (asyncRefresh || alreadyRunning) {
      dom.indexStateLabel.textContent = alreadyRunning ? "進行中の再索引を待っています…" : (result.message || "索引を更新中…");
      showToast(alreadyRunning ? "再索引はすでに進行中です。完了後に画面を更新します。" : "再索引を開始しました。完了後に画面を自動更新します。");
    } else {
      await Promise.allSettled([loadStatus(), loadDashboard(), loadTree(), loadItems()]);
      showToast("再索引が完了しました。");
    }
  } catch (error) {
    showToast(`再索引を開始できませんでした: ${error.message}`, true);
    await loadStatus({ quiet: true });
  } finally {
    dom.refreshButton.disabled = state.statusWasBusy;
  }
}

async function loadDashboard() {
  const raw = await api("/api/dashboard");
  const review = raw.review || {};
  for (const [key, metric] of [["required", "required_total"], ["pending", "required_pending"],
    ["certified", "certified_total"], ["regeneration", "regeneration_pending"],
    ["optional", "optional_unreviewed"], ["rejected", "rejected_total"]]) {
    dom[toCamel(`metric-review-${key}`)].textContent = formatNumber(review[metric] || 0);
  }
  dom.metricReviewCertifiedSub.textContent = `人手編集 ${formatNumber(review.edited_certified_total || 0)}件 ／ 音声・アライメント完了 ${formatNumber(review.certified_complete || 0)}件`;
  dom.metricReviewRegenerationSub.textContent = `修正版音声待ち ${formatNumber(review.regeneration_audio_pending || 0)}件 ／ アライメント待ち ${formatNumber(review.regeneration_alignment_pending || 0)}件（重複あり）`;
  const counts = raw.counts || raw.totals || raw;
  const totalAudio = readCount(counts, ["audio", "audios", "dialogue_audio", "audio_count", "num_audio"]);
  const totalAudioDuration = readCount(counts, ["audio_duration_seconds", "audio_seconds", "duration_seconds"]);
  const audioDurationKnown = readCount(counts, ["audio_duration_known", "duration_known"]);
  dom.metricTotalAudio.textContent = totalAudio === null ? "—" : formatNumber(totalAudio);
  dom.metricTotalAudioDuration.textContent = formatAudioDuration(totalAudioDuration);
  dom.metricTotalAudioSub.textContent = totalAudio !== null && audioDurationKnown !== null && audioDurationKnown < totalAudio
    ? `再生時間を取得済み: ${formatNumber(audioDurationKnown)} / ${formatNumber(totalAudio)}件`
    : "画像付き＋非画像の音声 UID 合計";
  const visualCounts = dashboardGroupCounts(raw, "visual", counts);
  const nonvisualCounts = dashboardGroupCounts(raw, "nonvisual", null);
  const visualImages = readCount(visualCounts, ["images", "image", "image_count", "num_images"]);
  const visualDialogues = readCount(visualCounts, ["dialogues", "dialogue", "dialogue_texts", "dialogue_count", "num_dialogues"]);
  const visualAudio = readCount(visualCounts, ["audio", "audios", "dialogue_audio", "audio_count", "num_audio"]);
  const visualAlignments = readCount(visualCounts, ["alignments", "alignment", "alignment_count", "num_alignments"]);
  const nonvisualDialogues = readCount(nonvisualCounts, ["dialogues", "dialogue", "dialogue_texts", "dialogue_count", "num_dialogues"]);
  const nonvisualAudio = readCount(nonvisualCounts, ["audio", "audios", "dialogue_audio", "audio_count", "num_audio"]);
  const nonvisualAlignments = readCount(nonvisualCounts, ["alignments", "alignment", "alignment_count", "num_alignments"]);

  setMetric("visual", "images", visualImages, "画像付きの論理 UID");
  setMetric("visual", "dialogues", visualDialogues, pipelineSupplement(visualDialogues, visualImages, "画像"));
  setMetric("visual", "audio", visualAudio, pipelineSupplement(visualAudio, visualDialogues, "対話"));
  setMetric("visual", "alignments", visualAlignments, pipelineSupplement(visualAlignments, visualAudio, "音声"));
  setMetric("nonvisual", "dialogues", nonvisualDialogues, "画像なしの論理 UID");
  setMetric("nonvisual", "audio", nonvisualAudio, pipelineSupplement(nonvisualAudio, nonvisualDialogues, "対話"));
  setMetric("nonvisual", "alignments", nonvisualAlignments, pipelineSupplement(nonvisualAlignments, nonvisualAudio, "音声"));

  const totalImages = readCount(counts, ["images", "image", "image_count", "num_images"]);
  const logicalTotal = readCount(counts, ["logical_uid_count", "items", "total", "count"]);
  dom.allCount.textContent = logicalTotal === null ? (totalImages === null ? "—" : formatNumber(totalImages)) : formatNumber(logicalTotal);

  state.collectionStats.clear();
  for (const collection of raw.collections || []) {
    const id = String(collection.collection ?? collection.id ?? collection.name ?? "");
    if (!id) continue;
    state.collectionStats.set(id, {
      planned: asNumber(collection.planned ?? collection.eligible ?? collection.target),
      generated: asNumber(collection.generated ?? collection.dialogues ?? collection.completed),
    });
  }
  updateCollectionProgress();

  const updated = raw.updated_at || raw.last_updated || raw.indexed_at || raw.last_refresh;
  if (updated) dom.lastUpdated.textContent = `最終更新: ${formatDateTime(updated)}`;
  updateCategoryOptions(raw.categories || raw.category_counts || []);
}

function setMetric(group, key, value, supplement) {
  const valueElement = dom[toCamel(`metric-${group}-${key}`)];
  const subElement = dom[toCamel(`metric-${group}-${key}-sub`)];
  valueElement.textContent = value === null ? "—" : formatNumber(value);
  subElement.textContent = supplement || "論理 UID";
}

function pipelineSupplement(value, denominator, denominatorLabel) {
  if (value !== null && denominator !== null && denominator > 0) {
    return `${denominatorLabel}の ${Math.round((value / denominator) * 100)}%`;
  }
  return "論理 UID";
}

function dashboardGroupCounts(raw, groupId, legacyFallback) {
  const groups = raw.groups;
  let group = null;
  if (Array.isArray(groups)) {
    group = groups.find((entry) => String(entry?.id || entry?.name || "") === groupId) || null;
  } else if (groups && typeof groups === "object") {
    group = groups[groupId] || null;
  }
  if (group) return group.totals || group.counts || group;

  // Compatibility with catalogues created before grouped dashboard totals
  // were introduced.  The collection rows still let us separate the two
  // pipelines without rebuilding the index first.
  const collectionRows = Array.isArray(raw.collections) ? raw.collections : [];
  if (collectionRows.length) {
    const selected = collectionRows.filter((entry) => {
      const id = String(entry?.collection || entry?.id || entry?.name || "")
        .replaceAll("-", "_")
        .toLowerCase();
      return id.startsWith("nonvisual") === (groupId === "nonvisual");
    });
    return sumDashboardRows(selected);
  }
  return legacyFallback || {};
}

function sumDashboardRows(rows) {
  const totals = { items: 0, images: 0, dialogues: 0, audio: 0, alignments: 0 };
  for (const row of rows) {
    for (const key of Object.keys(totals)) {
      const aliases = key === "items" ? ["items", "total", "count"] : [key];
      totals[key] += readCount(row, aliases) || 0;
    }
  }
  totals.logical_uid_count = totals.items;
  return totals;
}

async function loadTree() {
  const raw = await api("/api/tree");
  const nodes = normalizeTree(raw);
  dom.annotationCollection.replaceChildren(createElement("option", "", "すべてのコレクション"));
  dom.annotationCollection.firstElementChild.value = "";
  const seenCollections = new Set();
  for (const node of nodes) {
    const collection = node.filters.collection;
    if (!collection || seenCollections.has(collection)) continue;
    seenCollections.add(collection);
    const option = createElement("option", "", COLLECTION_LABELS[collection] || node.label);
    option.value = collection;
    dom.annotationCollection.append(option);
  }
  dom.annotationCollection.value = state.filters.collection;
  dom.collectionTree.replaceChildren();
  if (!nodes.length) {
    const message = createElement("p", "tree-note", "コレクション情報がありません");
    message.style.marginTop = "14px";
    dom.collectionTree.append(message);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const node of nodes) fragment.append(renderTreeNode(node, 0));
  dom.collectionTree.append(fragment);
  updateTreeSelection();
  updateCategoryOptions(raw.categories || []);
}

function normalizeTree(raw) {
  let source = Array.isArray(raw) ? raw : (raw.tree || raw.collections || raw.items || raw.nodes || (raw.id ? [raw] : []));
  if (!Array.isArray(source) && source && typeof source === "object") {
    source = Object.entries(source).map(([name, value]) => {
      if (value && typeof value === "object" && !Array.isArray(value)) return { name, ...value };
      return { name, count: value };
    });
  }
  const nodes = (Array.isArray(source) ? source : []).map((node) => normalizeTreeNode(node, {}, 0)).filter(Boolean);
  return nodes.sort((a, b) => {
    const ai = COLLECTION_ORDER.indexOf(a.value);
    const bi = COLLECTION_ORDER.indexOf(b.value);
    if (ai !== -1 || bi !== -1) return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    return a.label.localeCompare(b.label, "ja");
  });
}

function normalizeTreeNode(value, parentFilters = {}, depth = 0) {
  if (typeof value === "string") value = { id: value, label: value };
  if (!value || typeof value !== "object") return null;
  const id = String(value.value ?? value.id ?? value.collection ?? value.path ?? value.name ?? value.label ?? "");
  const rawLabel = String(value.label ?? value.display_name ?? value.name ?? id);
  const label = COLLECTION_LABELS[id] || rawLabel || id || "名称なし";
  const filters = { ...parentFilters };
  if (value.filter && typeof value.filter === "object") Object.assign(filters, value.filter);
  if (value.collection !== undefined) filters.collection = String(value.collection);
  if (value.category !== undefined) filters.category = String(value.category);
  if (value.directory !== undefined) filters.directory = String(value.directory);
  if (depth === 0 && !filters.collection) filters.collection = id;
  else if (depth === 1 && !filters.category) filters.category = id;
  else if (depth >= 2 && !value.directory) {
    const directoryId = String(value.path ?? value.value ?? value.id ?? value.name ?? "");
    filters.directory = parentFilters.directory && !directoryId.startsWith(parentFilters.directory)
      ? `${parentFilters.directory}/${directoryId}`
      : directoryId;
  }
  const childrenSource = [value.children, value.categories, value.directories, value.nodes, value.items]
    .find((candidate) => Array.isArray(candidate) && candidate.length) || [];
  const children = Array.isArray(childrenSource)
    ? childrenSource.map((child) => normalizeTreeNode(child, filters, depth + 1)).filter(Boolean)
    : [];
  return {
    value: id,
    label,
    count: asNumber(value.count ?? value.total ?? value.image_count ?? value.items_count),
    eligible: asNumber(value.eligible ?? value.target ?? value.planned),
    generated: asNumber(value.generated ?? value.completed ?? value.dialogue_count),
    filters,
    children,
  };
}

function renderTreeNode(node, depth) {
  const hasChildren = node.children.length > 0;
  const wrapper = hasChildren ? document.createElement("details") : document.createElement("div");
  wrapper.className = "tree-group";
  if (hasChildren && depth === 0) wrapper.open = true;

  const line = hasChildren ? document.createElement("summary") : document.createElement("div");
  if (hasChildren) line.append(createElement("span", "tree-chevron", ""));
  else line.style.paddingLeft = "22px";

  const button = createElement("button", "tree-node-button");
  button.type = "button";
  button.dataset.collection = node.filters.collection || "";
  button.dataset.category = node.filters.category || "";
  button.dataset.directory = node.filters.directory || "";
  button.dataset.label = node.label;
  button.title = node.value || node.label;
  button.setAttribute("aria-label", `${node.label}${node.count === null ? "" : `、${formatNumber(node.count)}件`}`);
  button.append(createElement("span", "folder-icon", ""));
  button.append(createElement("span", "tree-label", node.label));
  button.append(createElement("span", "tree-count", node.count === null ? "" : formatNumber(node.count)));
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    selectTreeFilters(node.filters, node.label);
  });
  line.append(button);
  wrapper.append(line);

  const dashboardStats = depth === 0 ? state.collectionStats.get(node.value) : null;
  const planned = node.eligible ?? dashboardStats?.planned ?? null;
  const generated = node.generated ?? dashboardStats?.generated ?? null;
  if (planned !== null || generated !== null) {
    const parts = [];
    if (planned !== null) parts.push(`対象 ${formatNumber(planned)}`);
    if (generated !== null) parts.push(`対話 ${formatNumber(generated)}`);
    wrapper.append(createElement("span", "tree-note collection-progress", parts.join(" / ")));
  }

  if (hasChildren) {
    const children = createElement("div", "tree-children");
    for (const child of node.children) children.append(renderTreeNode(child, depth + 1));
    wrapper.append(children);
  }
  return wrapper;
}

function updateCollectionProgress() {
  document.querySelectorAll(".tree-node-button").forEach((button) => {
    if (button.dataset.category || button.dataset.directory) return;
    const stats = state.collectionStats.get(button.dataset.collection);
    if (!stats) return;
    const wrapper = button.closest(".tree-group");
    if (!wrapper) return;
    const existing = Array.from(wrapper.children).find((child) => child.classList?.contains("collection-progress"));
    const parts = [];
    if (stats.planned !== null) parts.push(`対象 ${formatNumber(stats.planned)}`);
    if (stats.generated !== null) parts.push(`対話 ${formatNumber(stats.generated)}`);
    if (!parts.length) return;
    if (existing) existing.textContent = parts.join(" / ");
    else wrapper.insertBefore(createElement("span", "tree-note collection-progress", parts.join(" / ")), wrapper.children[1] || null);
  });
}

function selectTreeFilters(filters, label) {
  state.filters.collection = filters.collection || "";
  state.filters.category = filters.category || "";
  state.filters.directory = filters.directory || "";
  state.page = 1;
  dom.categoryFilter.value = state.filters.category;
  dom.annotationCollection.value = state.filters.collection;
  dom.activeCollection.textContent = label || state.filters.collection || "ALL DATA";
  updateTreeSelection();
  closeTree();
  void loadItems({ selectFirst: true });
}

function updateTreeSelection() {
  const allSelected = !state.filters.collection && !state.filters.category && !state.filters.directory;
  dom.allCollections.classList.toggle("is-selected", allSelected);
  dom.allCollections.setAttribute("aria-current", allSelected ? "true" : "false");
  document.querySelectorAll(".tree-node-button").forEach((button) => {
    const selected = button.dataset.collection === state.filters.collection
      && button.dataset.category === state.filters.category
      && button.dataset.directory === state.filters.directory;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-current", selected ? "true" : "false");
  });
}

async function loadItems({ scrollTop = false, selectFirst = false } = {}) {
  if (state.annotationMode && (state.filters.stage || state.filters.category || state.filters.directory)) {
    state.annotationAdvanced = true;
    updateAnnotationLayout();
  }
  if (state.listController) state.listController.abort();
  const controller = new AbortController();
  state.listController = controller;
  const requestId = ++state.listRequestId;
  setListState("loading");

  const params = filterParams();
  params.set("page", String(state.page));
  params.set("limit", String(state.limit));
  // offset is sent as well for APIs that expose offset/limit pagination.
  params.set("offset", String((state.page - 1) * state.limit));
  try {
    const raw = await api(`/api/items?${params.toString()}`, { signal: controller.signal });
    if (requestId !== state.listRequestId) return;
    const payload = !Array.isArray(raw) && raw.data && !Array.isArray(raw.data) ? raw.data : raw;
    const items = Array.isArray(payload) ? payload : (payload.items || payload.results || payload.data || []);
    state.items = Array.isArray(items) ? items.map(normalizeItem) : [];
    state.total = asNumber(payload.total ?? payload.count ?? payload.total_items) ?? state.items.length;
    state.pages = Math.max(1, asNumber(payload.pages ?? payload.total_pages) ?? Math.ceil(state.total / state.limit));
    state.page = Math.min(Math.max(1, asNumber(payload.page) ?? state.page), state.pages);

    updateCategoryOptions(payload.categories || payload.category_counts || []);
    renderItems();
    updatePagination();
    setListState(state.items.length ? "ready" : "empty");
    if (scrollTop) dom.datasetList.scrollTop = 0;
    if (state.annotationMode && state.items.length && (selectFirst || !state.selectedToken)
        && !state.editingDialogue && !state.savingDialogue && !state.reviewBusy) {
      await selectItem(state.items[0].token);
    }
    prefetchNextDetails();
  } catch (error) {
    if (error.name === "AbortError" || requestId !== state.listRequestId) return;
    dom.listErrorMessage.textContent = error.message;
    setListState("error");
  }
}

function normalizeItem(raw) {
  const token = String(raw.token ?? raw.item_token ?? raw.id ?? raw.uid ?? "");
  const uid = String(raw.uid ?? raw.source_uid ?? raw.id ?? token ?? "UIDなし");
  const item = {
    ...raw,
    token,
    uid,
    category: String(raw.category ?? raw.task_mode ?? raw.kind ?? "未分類"),
    collection: String(raw.collection ?? raw.dataset ?? raw.source ?? ""),
    path: String(raw.directory ?? raw.path ?? raw.image_path ?? raw.image ?? ""),
  };
  item.stageStatus = pipelineStatus(item);
  item.auxiliaryTask = normalizeAuxiliaryTask(
    raw.auxiliary_task ?? raw.auxiliaryTask ?? raw.summary?.auxiliary_task
      ?? raw.metadata?.auxiliary_task ?? raw.metadata?.auxiliaryTask,
  );
  item.reviewState = normalizeReview(item);
  return item;
}

function normalizeAuxiliaryTask(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const kind = String(raw.kind ?? raw.type ?? raw.task ?? "none").toLowerCase();
  if (!['counting', 'ocr'].includes(kind)) return null;
  return {
    ...raw,
    kind,
    grounding_source: String(raw.grounding_source ?? raw.source ?? ""),
    verification_status: String(raw.verification_status ?? ""),
  };
}

function normalizeReview(item) {
  const raw = item.review ?? item.summary?.review ?? item.metadata?.review;
  const review = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  const auxiliary = item.auxiliaryTask;
  const verification = String(
    item.verification_status ?? item.summary?.verification_status
      ?? item.metadata?.verification_status ?? auxiliary?.verification_status ?? "",
  );
  const explicitRequired = booleanValue(
    review.required ?? review.review_required ?? item.review_required
      ?? item.summary?.review_required ?? item.metadata?.review_required,
  );
  const inferredRequired = Boolean(
    auxiliary
      && auxiliary.grounding_source === "vlm_visual"
      && (verification === "unverified_vlm_visual" || !verification),
  );
  const required = explicitRequired ?? inferredRequired;
  let status = normalizeReviewStatus(
    review.status ?? review.decision ?? item.review_status ?? item.summary?.review_status,
  );
  if (required && status === "not_required") status = "pending";
  else if (!required && status === "pending") status = "not_required";
  const explicitUsable = booleanValue(
    review.dataset_usable ?? review.usable ?? item.dataset_usable
      ?? item.summary?.dataset_usable ?? item.metadata?.dataset_usable,
  );
  return {
    ...review,
    required,
    status,
    dataset_usable: explicitUsable ?? (status !== "rejected" && (!required || status === "approved")),
    content_sha256: String(
      review.content_sha256 ?? item.content_sha256 ?? item.dialogue_content_sha256
        ?? item.review_id ?? item.summary?.content_sha256 ?? item.summary?.review_id
        ?? item.metadata?.content_sha256 ?? "",
    ),
    note: String(review.note ?? review.review_note ?? ""),
    verification_status: verification,
  };
}

function normalizeReviewStatus(value) {
  const status = String(value ?? "").trim().toLowerCase();
  if (["approved", "approve", "accepted"].includes(status)) return "approved";
  if (["rejected", "reject", "denied"].includes(status)) return "rejected";
  if (["not_required", "not-required", "notrequired", "none"].includes(status)) return "not_required";
  return "pending";
}

function renderItems() {
  const fragment = document.createDocumentFragment();
  for (const item of state.items) {
    const row = createElement("button", "dataset-item");
    row.type = "button";
    row.dataset.token = item.token;
    row.classList.toggle("is-selected", item.token === state.selectedToken);
    row.setAttribute("aria-current", item.token === state.selectedToken ? "true" : "false");
    const reviewInfo = reviewStatusInfo(item.reviewState.status);
    row.setAttribute(
      "aria-label",
      `${item.uid} の詳細を表示${item.reviewState.required ? `、レビュー状態: ${reviewInfo.label}` : ""}`,
    );

    const thumb = createElement("span", "item-thumb");
    if (item.stageStatus.image.state === "complete") {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.decoding = "async";
      image.alt = "";
      image.src = mediaUrl(item, "image");
      image.addEventListener("error", () => {
        image.remove();
        thumb.classList.add("is-missing");
      }, { once: true });
      thumb.append(image);
    } else {
      thumb.classList.add("is-missing");
    }

    const main = createElement("span", "item-main");
    const top = createElement("span", "item-topline");
    top.append(createElement("span", "item-uid", item.uid));
    top.append(createElement("span", "category-badge", item.category));
    if (item.stageStatus.dialogue.state === "complete") top.append(renderReviewBadge(item.reviewState));
    if (item.regeneration_pending) top.append(createElement("span", "review-badge is-regeneration", "再作成待ち"));
    main.append(top);
    main.append(createElement("span", "item-path", [item.collection, item.path].filter(Boolean).join(" / ") || "保存先情報なし"));
    main.append(renderPipeline(item.stageStatus, false));
    row.append(thumb, main);
    row.addEventListener("click", () => void selectItem(item.token));
    fragment.append(row);
  }
  dom.datasetList.replaceChildren(fragment);
  dom.datasetList.setAttribute("aria-busy", "false");
  dom.resultCount.textContent = `${formatNumber(state.total)}件（論理 UID）`;
}

function renderReviewBadge(review) {
  const info = reviewStatusInfo(review?.status);
  const badge = createElement("span", `review-badge is-${info.key}`, info.label);
  badge.title = review?.dataset_usable ? "データセット利用可" : "現在はデータセット利用不可";
  return badge;
}

function reviewStatusInfo(status) {
  const key = normalizeReviewStatus(status);
  return {
    pending: { key, label: "要確認" },
    approved: { key, label: "認証済み（人手承認）" },
    rejected: { key, label: "却下" },
    not_required: { key, label: "未確認・承認任意" },
  }[key];
}

function pipelineStatus(item) {
  const result = {};
  let previousComplete = true;
  const nonvisual = ["nonvisual_v1", "nonvisual-v1", "nonvisual"].includes(
    String(item.collection || item.category || "").toLowerCase(),
  );
  for (const stage of STAGES) {
    if (stage.key === "image" && nonvisual) {
      result[stage.key] = { state: "excluded", label: stageStatusLabel("excluded") };
      // Image is intentionally not an upstream requirement for this collection.
      continue;
    }
    const explicit = explicitStageValue(item, stage.key);
    let status;
    if (explicit !== undefined && explicit !== null) status = normalizeStageStatus(explicit);
    else status = inferStageStatus(item, stage.key);

    if (status === "missing" && !previousComplete) status = "waiting";
    result[stage.key] = {
      state: status,
      label: stageStatusLabel(status),
    };
    previousComplete = previousComplete && status === "complete";
  }
  return result;
}

function explicitStageValue(item, key) {
  const aliases = {
    image: ["image"],
    dialogue: ["dialogue", "dialogue_text", "text"],
    audio: ["audio", "dialogue_audio", "speech"],
    alignment: ["alignment", "aligned"],
  }[key];
  const holders = [item.stages, item.stage_status, item.pipeline, item.availability, item.statuses];
  for (const holder of holders) {
    if (!holder || typeof holder !== "object") continue;
    for (const alias of aliases) {
      if (Object.prototype.hasOwnProperty.call(holder, alias)) return holder[alias];
    }
  }
  for (const alias of aliases) {
    const candidates = [`has_${alias}`, `${alias}_available`, `${alias}_exists`, `${alias}_status`];
    for (const candidate of candidates) {
      if (Object.prototype.hasOwnProperty.call(item, candidate)) return item[candidate];
    }
  }
  return undefined;
}

function inferStageStatus(item, key) {
  if (key === "image") return (item.image || item.image_path || item.image_url) ? "complete" : "missing";
  if (key === "dialogue") {
    const value = item.dialogue || item.dialogue_text || item.turns || item.messages;
    return (Array.isArray(value) ? value.length > 0 : Boolean(value)) ? "complete" : "missing";
  }
  if (key === "audio") return (item.audio || item.audio_path || item.audio_url || item.wav) ? "complete" : "missing";
  const value = item.alignment || item.alignment_path || item.alignments;
  return (Array.isArray(value) ? value.length > 0 : Boolean(value)) ? "complete" : "missing";
}

function normalizeStageStatus(value) {
  if (value === true || value === 1) return "complete";
  if (value === false || value === 0) return "missing";
  if (value && typeof value === "object") {
    return normalizeStageStatus(value.status ?? value.state ?? value.exists ?? value.complete ?? Boolean(value.path));
  }
  const text = String(value ?? "").toLowerCase();
  if (["complete", "completed", "ready", "exists", "available", "done", "ok", "true", "generated"].includes(text)) return "complete";
  if (["excluded", "not_applicable", "not-applicable", "na", "n/a", "対象外"].includes(text)) return "excluded";
  if (["waiting", "blocked", "pending_previous", "upstream", "前段階待ち"].includes(text)) return "waiting";
  return "missing";
}

function stageStatusLabel(status) {
  if (status === "complete") return "✓ 完了";
  if (status === "excluded") return "対象外";
  if (status === "waiting") return "前段階待ち";
  return "未生成";
}

function renderPipeline(statuses, detail) {
  const stepper = createElement("span", detail ? "pipeline-stepper detail-pipeline" : "pipeline-stepper");
  for (const stage of STAGES) {
    const status = statuses[stage.key] || { state: "missing", label: "未生成" };
    const step = createElement("span", `pipeline-step is-${status.state}`);
    step.title = `${stage.label}: ${status.label}`;
    step.setAttribute("aria-label", `${stage.label}: ${status.label}`);
    const dotText = status.state === "complete" ? "✓" : status.state === "excluded" ? "—" : status.state === "waiting" ? "…" : "!";
    step.append(createElement("span", "step-dot", dotText));
    const labels = createElement("span", "step-copy");
    labels.append(createElement("span", "step-label", stage.label));
    labels.append(createElement("span", "step-state-text", status.label));
    step.append(labels);
    stepper.append(step);
  }
  return stepper;
}

function updatePagination() {
  dom.pageCurrent.textContent = String(state.page);
  dom.pageTotal.textContent = String(state.pages);
  dom.pagePrev.disabled = state.page <= 1;
  dom.pageNext.disabled = state.page >= state.pages;
}

function setListState(mode) {
  dom.listLoading.hidden = mode !== "loading";
  dom.listEmpty.hidden = mode !== "empty";
  dom.listError.hidden = mode !== "error";
  dom.datasetList.setAttribute("aria-busy", mode === "loading" ? "true" : "false");
}

async function selectItem(token) {
  if (!token) return;
  if (state.savingDialogue || state.reviewBusy) return;
  if (state.editingDialogue) {
    if (token === state.selectedToken) return;
    if (!window.confirm("未保存の編集を取り消して別の対話を開きますか？")) return;
    cancelDialogueEdit();
  }
  const hadSelection = Boolean(state.selectedToken);
  state.selectedToken = token;
  updateSelectedRow();
  openDetail();
  await loadDetail(token, { scrollToStart: state.annotationMode && hadSelection, usePrefetch: true });
}

function detailParams() {
  const params = filterParams();
  params.set("page", String(state.page));
  params.set("limit", String(state.limit));
  return params;
}

function detailScope() {
  return detailParams().toString();
}

function clearDetailPrefetch(token = null) {
  for (const [key, entry] of state.detailPrefetch) {
    if (token !== null && entry.token !== token) continue;
    if (!entry.inUse) entry.controller.abort();
    state.detailPrefetch.delete(key);
  }
}

function warmDetailImage(raw, entry) {
  const detail = normalizeItem(raw.item || raw.data || raw);
  if (typeof Image === "undefined" || detail.stageStatus.image.state !== "complete") return;
  const image = new Image();
  image.decoding = "async";
  image.src = mediaUrl(detail, "image");
  entry.image = image;
  if (image.decode) void image.decode().catch(() => {});
}

function prefetchDetail(token) {
  const scope = detailScope();
  const key = `${scope}\n${token}`;
  const existing = state.detailPrefetch.get(key);
  if (existing && Date.now() - existing.createdAt < DETAIL_PREFETCH_TTL_MS) return existing;
  if (existing) clearDetailPrefetch(token);
  const controller = new AbortController();
  const entry = { token, scope, controller, createdAt: Date.now(), inUse: false };
  entry.promise = api(`/api/items/${encodeURIComponent(token)}?${scope}`, {
    signal: controller.signal, priority: "low",
  }).then(raw => {
    entry.raw = raw;
    warmDetailImage(raw, entry);
    return raw;
  }).catch(error => {
    if (state.detailPrefetch.get(key) === entry) state.detailPrefetch.delete(key);
    throw error;
  });
  // Prefetch failures are retried by normal navigation, never surfaced as save failures.
  void entry.promise.catch(() => {});
  state.detailPrefetch.set(key, entry);
  for (const [oldKey, oldEntry] of state.detailPrefetch) {
    if (state.detailPrefetch.size <= DETAIL_PREFETCH_LIMIT) break;
    if (oldKey !== key && !oldEntry.inUse) {
      oldEntry.controller.abort();
      state.detailPrefetch.delete(oldKey);
    }
  }
  return entry;
}

function prefetchNextDetails() {
  if (!state.annotationMode || document.hidden || !state.selectedItem) return;
  const scope = detailScope();
  if (state.reviewQueueScope !== scope) {
    state.reviewQueueScope = scope;
    state.excludedReviewTokens.clear();
    clearDetailPrefetch();
  }
  const next = state.navigation.next || listNeighbor(state.selectedToken, 1);
  const afterNext = next && listNeighbor(next, 1);
  for (const token of new Set([next, afterNext])) {
    if (token && token !== state.selectedToken && !state.excludedReviewTokens.has(token)) prefetchDetail(token);
  }
}

async function loadDetail(token, { scrollToStart = false, usePrefetch = false } = {}) {
  if (state.editingDialogue) return; // Automatic refresh must not replace a draft.
  if (state.detailController) state.detailController.abort();
  const controller = new AbortController();
  state.detailController = controller;
  const requestId = ++state.detailRequestId;
  // A background refresh must not collapse a long, page-height transcript.
  const keepVisible = state.annotationMode && !dom.detailContent.hidden && state.selectedItem?.token === token;
  dom.detailLoading.hidden = keepVisible;
  dom.detailEmpty.hidden = true;
  if (!keepVisible) dom.detailContent.hidden = true;
  const scope = detailScope();
  const key = `${scope}\n${token}`;
  const prefetched = usePrefetch ? state.detailPrefetch.get(key) : null;
  const reusable = prefetched && Date.now() - prefetched.createdAt < DETAIL_PREFETCH_TTL_MS;
  try {
    let raw;
    if (reusable) {
      prefetched.inUse = true;
      try { raw = await prefetched.promise; }
      catch (error) {
        if (controller.signal.aborted) return;
        raw = await api(`/api/items/${encodeURIComponent(token)}?${scope}`, { signal: controller.signal });
      }
    } else raw = await api(`/api/items/${encodeURIComponent(token)}?${scope}`, { signal: controller.signal });
    if (requestId !== state.detailRequestId || controller.signal.aborted || state.editingDialogue) return;
    const source = raw.item || raw.data || raw;
    const detail = normalizeItem(source);
    detail.token = detail.token || token;
    const navigation = raw.navigation || source.navigation || {};
    state.navigation.previous = navigationToken(navigation.previous ?? navigation.prev ?? raw.previous_token ?? source.previous_token) || listNeighbor(token, -1);
    state.navigation.next = navigationToken(navigation.next ?? raw.next_token ?? source.next_token) || listNeighbor(token, 1);
    for (const direction of ["previous", "next"]) {
      if (state.reviewQueueScope === scope && state.excludedReviewTokens.has(state.navigation[direction])) {
        state.navigation[direction] = listNeighbor(token, direction === "previous" ? -1 : 1);
      }
    }
    renderDetail(detail);
    dom.detailContent.hidden = false;
    scheduleDialogueEditorSizing();
    if (scrollToStart && state.annotationMode) dom.detailPanel.scrollIntoView({ block: "start", behavior: "auto" });
    prefetchNextDetails();
  } catch (error) {
    if (error.name === "AbortError" || requestId !== state.detailRequestId) return;
    if (!keepVisible) {
      dom.detailEmpty.hidden = false;
      dom.detailEmpty.querySelector("h2").textContent = "詳細を取得できませんでした";
      dom.detailEmpty.querySelector("p").textContent = error.message;
    }
    showToast(`詳細を取得できませんでした: ${error.message}`, true);
  } finally {
    if (prefetched) {
      prefetched.inUse = false;
      if (state.detailPrefetch.get(key) === prefetched) state.detailPrefetch.delete(key);
    }
    if (requestId === state.detailRequestId) dom.detailLoading.hidden = true;
  }
}

function renderDetail(item) {
  const changedItem = state.selectedItem?.token !== item.token;
  state.selectedToken = item.token;
  state.selectedItem = item;
  updateSelectedRow();
  dom.detailCategory.textContent = item.category.toUpperCase();
  dom.detailUid.textContent = item.uid;
  dom.detailUid.title = item.uid;
  dom.detailPath.textContent = [item.collection, item.path].filter(Boolean).join(" / ") || "保存先情報なし";
  dom.detailStepper.replaceChildren(renderPipeline(item.stageStatus, true));
  dom.itemPrev.disabled = !state.navigation.previous;
  dom.itemNext.disabled = !state.navigation.next;

  renderAvailability("image", item.stageStatus.image);
  renderAvailability("dialogue", item.stageStatus.dialogue);
  renderAvailability("audio", item.stageStatus.audio);
  renderAvailability("alignment", item.stageStatus.alignment);
  renderDetailImage(item);
  renderReview(item);
  renderAiSuggestion(item);
  renderDialogue(item, { resetScroll: changedItem });
  renderAudio(item);
  renderAlignment(item);
  renderMetadata(item);
}

function renderAvailability(key, status) {
  const element = dom[toCamel(`${key}-availability`)];
  element.className = "availability";
  element.classList.add(status.state === "complete" ? "is-ready" : status.state === "excluded" ? "is-na" : "is-missing");
  element.textContent = status.label;
}

function renderDetailImage(item) {
  const ready = item.stageStatus.image.state === "complete";
  dom.annotationImageOpen.hidden = !ready;
  dom.annotationImageOpen.removeAttribute("href");
  dom.detailImage.hidden = !ready;
  dom.imageMissing.hidden = ready;
  dom.detailImage.removeAttribute("src");
  if (!ready) return;
  dom.detailImage.alt = `${item.uid} の画像`;
  dom.detailImage.src = mediaUrl(item, "image");
  dom.annotationImageOpen.href = dom.detailImage.src;
  dom.detailImage.onerror = () => {
    dom.detailImage.hidden = true;
    dom.annotationImageOpen.hidden = true;
    dom.imageMissing.hidden = false;
    dom.imageMissing.textContent = "画像ファイルを読み込めませんでした";
  };
}

function renderReview(item) {
  const review = item.reviewState;
  const auxiliary = item.auxiliaryTask || {};
  const visible = Boolean(item.review_available ?? normalizeDialogue(item.dialogue).length);
  dom.reviewSection.hidden = !visible;
  dom.reviewFacts.replaceChildren();
  dom.reviewQa.replaceChildren();
  dom.reviewError.hidden = true;
  dom.reviewError.textContent = "";
  dom.reviewSection.removeAttribute("aria-busy");
  if (!visible) return;

  const status = reviewStatusInfo(review.status);
  dom.reviewStatusBadge.className = `review-status-badge is-${status.key}`;
  dom.reviewStatusBadge.textContent = status.label;
  dom.reviewNote.value = review.note;
  dom.reviewApprove.textContent = activeAiSuggestion(item) ? "AI修正案を承認して保存" : "承認して利用する";
  dom.reviewDescription.textContent = review.required
    ? "人間の承認が必須です。画像と対話全文を確認してください。未確認・却下の間は音声生成に進めません。"
    : "承認は任意です。未確認でも利用できますが、対話全体を確認して承認すると認証ラベルが付きます。本文も編集できます。";

  const facts = [
    ["補助タスク", auxiliaryKindLabel(auxiliary.kind)],
    ["根拠", groundingSourceLabel(auxiliary.grounding_source)],
  ];
  if (auxiliary.kind === "counting") {
    if (auxiliary.counted_target) facts.push(["数えた対象", auxiliary.counted_target]);
    if (auxiliary.answer_count !== undefined && auxiliary.answer_count !== null) {
      facts.push([activeAiSuggestion(item) ? "AI修正案の回答" : "VLMの回答", `${auxiliary.answer_count}個`]);
    }
  } else if (auxiliary.recognized_text) {
    facts.push([activeAiSuggestion(item) ? "AI修正案の文字" : "VLMが読んだ文字", auxiliary.recognized_text]);
  }
  if (review.verification_status) {
    facts.push(["検証状態", verificationStatusLabel(review.verification_status)]);
  }
  if (item.human_revision) facts.push(["修正版", `人手編集済み（${formatDateTime(item.human_revision.edited_at)}）`]);
  for (const [label, value] of facts) {
    dom.reviewFacts.append(createElement("dt", "", label));
    dom.reviewFacts.append(createElement("dd", "", value || "—"));
  }

  renderReviewQa(item);
  dom.reviewUsage.className = `review-usage is-${status.key}`;
  dom.reviewUsage.textContent = review.status === "approved"
    ? (item.regeneration_pending
      ? "✓ 人手認証済み。編集後の音声・アライメント再作成待ちです。古い音声は学習から除外します。"
      : "✓ 人手認証済み。必要な音声・アライメントが揃った後に学習へ採用します。")
    : status.key === "rejected"
      ? "却下済みのため、学習には使用しません。データは後の一括再生成用に保持されます。"
      : review.required
        ? "承認されるまで、音声生成・学習への採用を保留します。"
        : "未確認ですが、承認必須ではないため音声生成・学習に利用できます。";
  setReviewButtonsBusy(false);
  dom.reviewReset.hidden = status.key === "pending" || status.key === "not_required";
}

function renderReviewQa(item) {
  const turns = normalizeDialogue(
    item.dialogue ?? item.turns ?? item.messages ?? item.dialogue_text ?? item.metadata?.dialogue,
  );
  const auxiliary = item.auxiliaryTask;
  const questionIndex = integerValue(auxiliary?.question_turn_index);
  const answerIndex = integerValue(auxiliary?.answer_turn_index);
  const entries = [
    ["確認対象の質問", questionIndex === null ? null : turns[questionIndex]],
    [activeAiSuggestion(item) ? "AI修正案の回答" : "VLMの回答", answerIndex === null ? null : turns[answerIndex]],
  ].filter(([, turn]) => turn);
  if (!entries.length) {
    dom.reviewQa.append(createElement("p", "review-qa-missing", "該当する質問・回答の位置情報がありません。対話全体を確認してください。"));
    return;
  }
  for (const [label, turn] of entries) {
    const row = createElement("div", "review-qa-row");
    row.append(createElement("span", "review-qa-label", label));
    row.append(createElement("p", "review-qa-text", turn.text || "（空の発話）"));
    dom.reviewQa.append(row);
  }
}

function auxiliaryKindLabel(kind) {
  if (kind === "counting") return "カウント";
  if (kind === "ocr") return "日本語OCR";
  return String(kind || "—");
}

function groundingSourceLabel(source) {
  if (source === "vlm_visual") return "正解注釈なし・VLMの画像判断";
  if (source === "canonical") return "正解注釈あり";
  return String(source || "—");
}

function verificationStatusLabel(status) {
  const labels = {
    unverified_vlm_visual: "未確認（VLMの画像判断）",
    verified_canonical: "正解注釈で確認済み",
    approved: "人手承認済み",
    rejected: "人手確認で却下",
  };
  return labels[status] || String(status).replaceAll("_", " ");
}

function activeAiSuggestion(item) {
  return item?.reviewState?.required && item.reviewState.status === "pending"
    ? item.ai_suggestion : null;
}

function renderAiSuggestion(item) {
  const suggestion = activeAiSuggestion(item);
  dom.aiSuggestion.hidden = !suggestion;
  dom.aiSuggestion.open = false;
  dom.aiSuggestionConcerns.replaceChildren();
  dom.aiSuggestionConcerns.hidden = !suggestion?.concerns?.length;
  dom.aiSuggestionDiff.replaceChildren();
  if (!suggestion) return;
  const label = suggestion.verdict === "unchanged" ? "AI事前確認（未承認・変更なし）"
    : suggestion.verdict === "needs_attention" ? "AI事前修正（未承認・要確認）" : "AI事前修正（未承認）";
  dom.aiSuggestionLabel.textContent = `${label}・元の対話と差分`;
  for (const concern of suggestion.concerns || []) {
    dom.aiSuggestionConcerns.append(createElement("p", "review-error", `要確認: ${concern}`));
  }
  const proposed = normalizeDialogue(item.dialogue);
  const original = normalizeDialogue(item.original_dialogue);
  for (const [index, before] of original.entries()) {
    const after = proposed[index];
    const changed = after && before.text !== after.text;
    const block = createElement("div", "ai-suggestion-turn");
    block.append(createElement("p", "message-speaker", `${index + 1}. ${before.speaker}（${changed ? "変更あり" : "変更なし"}）`));
    block.append(createElement("p", changed ? "ai-suggestion-before" : "", `元の本文: ${before.text}`));
    if (changed) block.append(createElement("p", "ai-suggestion-after", `修正案: ${after.text}`));
    for (const change of suggestion.changes || []) {
      if (change.turn_index === index && change.reason) {
        block.append(createElement("p", "ai-suggestion-reason", `修正理由: ${change.reason}`));
      }
    }
    dom.aiSuggestionDiff.append(block);
  }
  for (const [key, label] of [["answer_count", "正解の数"], ["counted_target", "数えた対象"], ["recognized_text", "読み取り文字"]]) {
    const before = item.original_auxiliary_task?.[key], after = item.auxiliaryTask?.[key];
    if (before !== after) dom.aiSuggestionDiff.append(createElement("p", "ai-suggestion-turn", `${label}: ${before ?? "—"} → ${after ?? "—"}`));
  }
}

async function submitReview(decision) {
  const item = state.selectedItem;
  if (!item?.token || !item.review_available || state.reviewBusy || state.editingDialogue) return;
  const nextToken = state.annotationMode && dom.annotationAutoNext.checked && decision !== "pending" ? state.navigation.next : "";
  const queueScope = detailScope();
  let succeeded = false;
  state.reviewBusy = true;
  const review = item.reviewState;
  const suggestion = activeAiSuggestion(item);
  const approveSuggestion = decision === "approved" && suggestion;
  dom.reviewError.hidden = true;
  dom.reviewError.textContent = "";
  setReviewButtonsBusy(true);
  dom.reviewSection.setAttribute("aria-busy", "true");
  try {
    const endpoint = approveSuggestion ? `/api/items/${encodeURIComponent(item.token)}/dialogue`
      : `/api/items/${encodeURIComponent(item.token)}/review`;
    const result = await api(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision,
        content_sha256: review.content_sha256,
        // `fingerprint` is the backend's content-bound concurrency key;
        // keep the descriptive alias for clients and future API versions.
        fingerprint: review.content_sha256,
        note: dom.reviewNote.value.trim(),
        ...(approveSuggestion ? {
          ai_suggestion_id: suggestion.suggestion_id,
          dialogue: normalizeDialogue(item.dialogue).map(turn => ({ text: turn.text })),
          auxiliary_task: item.auxiliaryTask,
        } : {}),
      }),
    });
    const returnedItem = result.item || result.data?.item || {};
    const returnedReview = result.review || result.data?.review
      || (result.status || result.decision ? result : null);
    const datasetUsable = booleanValue(
      result.dataset_usable ?? result.data?.dataset_usable ?? returnedItem.dataset_usable,
    ) ?? (decision !== "rejected" && (!review.required || decision === "approved"));
    const status = decision === "pending" && !review.required ? "not_required" : decision;
    const next = normalizeItem({
      ...item,
      ...(suggestion ? {
        ai_suggestion: null,
        dialogue: item.original_dialogue,
        auxiliary_task: item.original_auxiliary_task,
      } : {}),
      ...returnedItem,
      review_status: status,
      dataset_usable: datasetUsable,
      review: returnedReview ? {
        ...returnedReview,
        required: review.required,
        status: returnedReview.status ?? status,
        dataset_usable: datasetUsable,
      } : returnedItem.review || {
        ...review,
        status,
        dataset_usable: datasetUsable,
        note: dom.reviewNote.value.trim(),
      },
    });
    applyReviewedItemToQueue(next, queueScope);
    if (!nextToken || queueScope !== detailScope()) renderDetail(next);
    showToast(decision === "approved"
      ? approveSuggestion ? "表示中のAI修正案を保存し、人手承認しました。" : "承認しました。この対話は学習データセットで利用されます。"
      : decision === "rejected"
        ? "却下しました。この対話は学習データセットから除外されます。"
        : review.required ? "未確認に戻しました。音声生成・学習を保留します。" : "認証ラベルを外しました。通常データとして利用可能です。");
    succeeded = true;
  } catch (error) {
    dom.reviewError.textContent = `レビュー結果を保存できませんでした: ${error.message}`;
    dom.reviewError.hidden = false;
    showToast(dom.reviewError.textContent, true);
  } finally {
    state.reviewBusy = false;
    dom.reviewSection.removeAttribute("aria-busy");
    setReviewButtonsBusy(false);
  }
  if (succeeded) {
    // Durable save first, then the next record. Aggregate queries must never
    // sit on the annotation navigation's critical path.
    if (nextToken && queueScope === detailScope()) await selectItem(nextToken);
    refreshAfterAnnotation();
  }
}

function applyReviewedItemToQueue(item, scope) {
  clearDetailPrefetch(item.token);
  if (state.detailController) state.detailController.abort();
  state.detailRequestId += 1;
  // A pre-save list response must not resurrect a just-approved pending row.
  if (state.listController) state.listController.abort();
  state.listRequestId += 1;
  if (scope !== detailScope()) return;
  const wanted = state.filters.review_status;
  const actual = item.reviewState.status;
  const statusMatches = !wanted || (wanted === "unreviewed"
    ? ["pending", "not_required"].includes(actual) : wanted === actual);
  const remains = statusMatches
    && (!state.filters.review_required || item.reviewState.required)
    && (state.filters.stage !== "regeneration_pending" || item.regeneration_pending);
  if (state.reviewQueueScope !== scope) {
    state.reviewQueueScope = scope;
    state.excludedReviewTokens.clear();
  }
  if (!remains) {
    state.excludedReviewTokens.add(item.token);
    if (state.excludedReviewTokens.size > 100) state.excludedReviewTokens.delete(state.excludedReviewTokens.values().next().value);
  } else state.excludedReviewTokens.delete(item.token);
  const index = state.items.findIndex(row => row.token === item.token);
  if (index < 0) return;
  if (remains) state.items[index] = item;
  else {
    state.items.splice(index, 1);
    state.total = Math.max(0, state.total - 1);
    state.pages = Math.max(1, Math.ceil(state.total / state.limit));
    state.page = Math.min(state.page, state.pages);
  }
  renderItems();
  updatePagination();
  setListState(state.items.length ? "ready" : "empty");
}

let annotationRefreshTimer = null;
let annotationRefreshRunning = false;
let annotationRefreshPending = false;

function refreshAfterAnnotation() {
  // The saved row is patched immediately. Reconcile the queue and aggregates
  // together only when the annotator pauses, instead of issuing them per click.
  annotationRefreshPending = true;
  if (annotationRefreshTimer !== null) window.clearTimeout(annotationRefreshTimer);
  annotationRefreshTimer = window.setTimeout(flushAnnotationRefresh, 1000);
}

async function flushAnnotationRefresh() {
  annotationRefreshTimer = null;
  if (annotationRefreshRunning) return;
  if (state.reviewBusy || state.savingDialogue || state.editingDialogue) {
    annotationRefreshTimer = window.setTimeout(flushAnnotationRefresh, 1000);
    return;
  }
  annotationRefreshRunning = true;
  annotationRefreshPending = false;
  try {
    await Promise.allSettled([loadItems(), loadDashboard(), loadTree()]);
  } finally {
    annotationRefreshRunning = false;
    if (annotationRefreshPending) {
      annotationRefreshTimer = window.setTimeout(flushAnnotationRefresh, 1000);
    }
  }
}

function setReviewButtonsBusy(busy) {
  document.body.classList.toggle("annotation-editing", state.editingDialogue);
  for (const button of [dom.reviewApprove, dom.reviewReject, dom.reviewReset, dom.reviewEdit]) {
    button.disabled = busy || state.editingDialogue;
  }
  dom.reviewNote.disabled = busy;
}

function startDialogueEdit() {
  const item = state.selectedItem;
  if (!item?.review_available || state.editingDialogue) return;
  state.editingDialogue = true;
  dom.dialogueEditorActions.hidden = false;
  dom.dialogueHeadingActions.hidden = false;
  dom.dialogueEditorError.hidden = true;
  renderDialogue(item);
  setReviewButtonsBusy(false);
  dom.dialogueEditorFacts.replaceChildren();
  const aux = item.auxiliaryTask || {};
  const fields = aux.kind === "counting"
    ? [["counted_target", "数えた対象", "text"], ["answer_count", "正解の数", "number"]]
    : aux.kind === "ocr" ? [["recognized_text", "正しい読み取り文字", "text"]] : [];
  for (const [key, title, type] of fields) {
    const label = createElement("label", "review-note-label", title);
    const input = document.createElement("input");
    input.type = type; input.dataset.auxiliaryField = key;
    input.value = aux[key] ?? ""; input.required = true;
    if (type === "number") { input.min = "1"; input.max = "1000"; input.step = "1"; }
    else input.maxLength = 1000;
    label.append(input); dom.dialogueEditorFacts.append(label);
  }
  // Editing changes controls in place; it must not move the work panels.
}

function cancelDialogueEdit() {
  if (state.savingDialogue) return;
  state.editingDialogue = false;
  dom.dialogueEditorActions.hidden = true;
  dom.dialogueHeadingActions.hidden = true;
  if (state.selectedItem) renderDialogue(state.selectedItem);
  setReviewButtonsBusy(false);
}

async function saveDialogueEdit() {
  const item = state.selectedItem;
  if (!state.editingDialogue || state.savingDialogue || !item) return;
  const nextToken = state.annotationMode && dom.annotationAutoNext.checked ? state.navigation.next : "";
  const queueScope = detailScope();
  let succeeded = false;
  const inputs = [...dom.detailDialogue.querySelectorAll("textarea")];
  const facts = [...dom.dialogueEditorFacts.querySelectorAll("input")];
  if (![...inputs, ...facts].every((input) => input.reportValidity())) return;
  const auxiliary = { ...item.auxiliaryTask };
  for (const input of facts) auxiliary[input.dataset.auxiliaryField] = input.type === "number" ? Number(input.value) : input.value.trim();
  state.savingDialogue = true;
  dom.dialogueSave.disabled = dom.dialogueCancel.disabled = true;
  for (const input of [...inputs, ...facts]) input.disabled = true;
  dom.reviewNote.disabled = true;
  dom.dialogueEditorError.hidden = true;
  try {
    const result = await api(`/api/items/${encodeURIComponent(item.token)}/dialogue`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content_sha256: item.reviewState.content_sha256,
        dialogue: inputs.map((input) => ({ text: input.value.trim() })),
        auxiliary_task: ["counting", "ocr"].includes(auxiliary.kind) ? auxiliary : null,
        ...(activeAiSuggestion(item) ? { ai_suggestion_id: item.ai_suggestion.suggestion_id } : {}),
        note: dom.reviewNote.value.trim() }),
    });
    state.editingDialogue = false;
    dom.dialogueEditorActions.hidden = true;
    dom.dialogueHeadingActions.hidden = true;
    const updated = normalizeItem(result.item);
    applyReviewedItemToQueue(updated, queueScope);
    if (!nextToken || queueScope !== detailScope()) renderDetail(updated);
    showToast(result.content_changed === false
      ? "本文に変更はありません。人手承認の認証ラベルを付けました。"
      : "修正版を保存して承認しました。音声・アライメントは修正版から作成してください。");
    succeeded = true;
  } catch (error) {
    dom.dialogueEditorError.textContent = `修正版を保存できませんでした: ${error.message}`;
    dom.dialogueEditorError.hidden = false;
    showToast(dom.dialogueEditorError.textContent, true);
  } finally {
    state.savingDialogue = false;
    dom.dialogueSave.disabled = dom.dialogueCancel.disabled = false;
    for (const input of [...inputs, ...facts]) input.disabled = false;
    setReviewButtonsBusy(false);
  }
  if (succeeded) {
    if (nextToken && queueScope === detailScope()) await selectItem(nextToken);
    refreshAfterAnnotation();
  }
}

function renderDialogue(item, { resetScroll = false } = {}) {
  const previousScrollTop = resetScroll ? 0 : (dom.detailDialogue.scrollTop || 0);
  dom.detailDialogue.replaceChildren();
  const turns = normalizeDialogue(item.dialogue ?? item.turns ?? item.messages ?? item.dialogue_text ?? item.metadata?.dialogue);
  if (!turns.length) {
    dom.detailDialogue.append(createElement("p", "missing-block", "まだ対話テキストは作成されていません。"));
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const [index, turn] of turns.entries()) {
    const role = /assistant|system|model|agent|bot|アシスタント/i.test(turn.speaker) ? "assistant" : "user";
    const card = createElement("article", `message is-${role}`);
    const targetLabel = auxiliaryTurnLabel(item.auxiliaryTask, index);
    if (targetLabel) card.classList.add("is-review-target");
    const speaker = createElement("p", "message-speaker", displaySpeaker(turn.speaker, role));
    if (targetLabel) speaker.append(createElement("span", "message-review-label", targetLabel));
    if (activeAiSuggestion(item) && item.original_dialogue?.[index]?.text !== turn.text) {
      speaker.append(createElement("span", "message-ai-label", "AI修正案・未承認"));
    }
    card.append(speaker);
    if (state.editingDialogue) {
      const input = document.createElement("textarea");
      input.className = "dialogue-edit-text"; input.rows = 3;
      input.required = true; input.maxLength = 10000; input.value = turn.text;
      input.setAttribute("aria-label", `${index + 1}番目の発話：${displaySpeaker(turn.speaker, role)}`);
      input.addEventListener("input", () => { if (state.annotationMode) fitDialogueInput(input); });
      card.append(input);
    } else card.append(createElement("p", "message-text", turn.text || "（空の発話）"));
    fragment.append(card);
  }
  dom.detailDialogue.append(fragment);
  dom.detailDialogue.scrollTop = previousScrollTop;
  scheduleDialogueEditorSizing({ scrollTop: previousScrollTop });
}

function fitDialogueInput(input) {
  input.style.height = "auto";
  input.style.height = `${Math.max(input.scrollHeight + 2 || 0, 96)}px`;
}

function scheduleDialogueEditorSizing({ scrollTop = null } = {}) {
  const selectedToken = state.selectedItem?.token;
  const resize = () => {
    if (selectedToken !== state.selectedItem?.token) return;
    for (const input of dom.detailDialogue.querySelectorAll("textarea")) {
      if (state.annotationMode) fitDialogueInput(input);
      else input.style.height = "";
    }
    if (scrollTop !== null) dom.detailDialogue.scrollTop = scrollTop;
  };
  if (window.requestAnimationFrame) window.requestAnimationFrame(resize);
  else window.setTimeout(resize, 0);
}

function auxiliaryTurnLabel(auxiliary, index) {
  if (!auxiliary) return "";
  if (integerValue(auxiliary.question_turn_index) === index) {
    return auxiliary.kind === "counting" ? "カウント質問" : "OCR質問";
  }
  if (integerValue(auxiliary.answer_turn_index) === index) {
    return auxiliary.kind === "counting" ? "カウント回答" : "OCR回答";
  }
  return "";
}

function normalizeDialogue(raw) {
  if (typeof raw === "string") return [{ speaker: "assistant", text: raw }];
  if (!Array.isArray(raw)) return [];
  return raw.map((turn, index) => {
    if (typeof turn === "string") return { speaker: index % 2 ? "assistant" : "user", text: turn };
    return {
      speaker: String(turn?.speaker ?? turn?.role ?? turn?.channel ?? (index % 2 ? "assistant" : "user")),
      text: String(turn?.text ?? turn?.content ?? turn?.transcript ?? turn?.sentence ?? ""),
    };
  });
}

function displaySpeaker(speaker, role) {
  if (role === "assistant") return /assistant|model|agent|bot/i.test(speaker) ? "アシスタント" : speaker;
  return /user|human/i.test(speaker) ? "ユーザー" : speaker;
}

function renderAudio(item) {
  const ready = item.stageStatus.audio.state === "complete";
  dom.audioCard.hidden = !ready;
  dom.audioMissing.hidden = ready;
  dom.detailAudio.pause();
  dom.detailAudio.removeAttribute("src");
  dom.detailAudio.load();
  if (ready) {
    dom.detailAudio.src = mediaUrl(item, "audio");
    dom.detailAudio.load();
  }
}

function renderAlignment(item) {
  dom.alignmentBody.replaceChildren();
  const segments = normalizeAlignment(item.alignment ?? item.alignments ?? item.alignment_data ?? item.segments ?? item.metadata?.alignment);
  dom.alignmentWrap.hidden = !segments.length;
  dom.alignmentMissing.hidden = segments.length > 0;
  const fragment = document.createDocumentFragment();
  for (const segment of segments) {
    const row = document.createElement("tr");
    row.className = "alignment-row";
    row.tabIndex = 0;
    row.dataset.start = String(segment.start);
    row.dataset.end = String(segment.end);
    row.setAttribute("aria-label", `${formatTime(segment.start)}から再生: ${segment.text}`);
    row.append(createElement("td", "", formatTime(segment.start)));
    row.append(createElement("td", "", formatTime(segment.end)));
    row.append(createElement("td", "", displaySpeaker(segment.speaker, /assistant|model|agent/i.test(segment.speaker) ? "assistant" : "user")));
    row.append(createElement("td", "", segment.text || "—"));
    const seek = () => seekAudio(segment.start);
    row.addEventListener("click", seek);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        seek();
      }
    });
    fragment.append(row);
  }
  dom.alignmentBody.append(fragment);
}

function normalizeAlignment(raw) {
  if (raw && !Array.isArray(raw) && typeof raw === "object") raw = raw.segments || raw.items || raw.words || raw.alignment || [];
  if (!Array.isArray(raw)) return [];
  return raw.map((segment) => {
    if (!segment || typeof segment !== "object") return null;
    const start = secondsValue(segment.start ?? segment.start_sec ?? segment.begin ?? segment.start_time ?? segment.timestamp?.[0]);
    const end = secondsValue(segment.end ?? segment.end_sec ?? segment.finish ?? segment.end_time ?? segment.timestamp?.[1]);
    return {
      start,
      end: Math.max(start, end),
      speaker: String(segment.speaker ?? segment.role ?? segment.channel ?? "—"),
      text: String(segment.text ?? segment.content ?? segment.word ?? segment.token ?? ""),
    };
  }).filter(Boolean);
}

function renderMetadata(item) {
  dom.metadataGrid.replaceChildren();
  const skipped = new Set(["dialogue", "turns", "messages", "dialogue_text", "alignment", "alignments", "alignment_data", "segments", "stageStatus", "navigation"]);
  const entries = [];
  for (const [key, value] of Object.entries(item)) {
    if (skipped.has(key) || value === undefined || typeof value === "function") continue;
    if (key === "metadata" && value && typeof value === "object" && !Array.isArray(value)) {
      for (const [metaKey, metaValue] of Object.entries(value)) {
        if (!skipped.has(metaKey)) entries.push([metaKey, metaValue]);
      }
    } else {
      entries.push([key, value]);
    }
  }
  const seen = new Set();
  for (const [key, value] of entries) {
    if (seen.has(key)) continue;
    seen.add(key);
    dom.metadataGrid.append(createElement("dt", "", key));
    dom.metadataGrid.append(createElement("dd", "", displayValue(value)));
  }
  if (!entries.length) {
    dom.metadataGrid.append(createElement("dt", "", "情報"));
    dom.metadataGrid.append(createElement("dd", "", "追加メタデータはありません"));
  }
}

function displayValue(value) {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") {
    try {
      const text = JSON.stringify(value, null, 2);
      return text.length > 4000 ? `${text.slice(0, 4000)}\n…（省略）` : text;
    } catch (_) {
      return String(value);
    }
  }
  return String(value);
}

function mediaUrl(item, kind) {
  const direct = kind === "image"
    ? (item.image_url || item.media?.image)
    : (item.audio_url || item.media?.audio);
  const url = direct && isSafeLocalUrl(direct) ? String(direct)
    : `/api/items/${encodeURIComponent(item.token)}/media/${kind}`;
  if (kind !== "image") return url;
  // Browser caching must bind image bytes to the exact displayed review version.
  const revision = item.current_image_sha256 || item.summary?.current_image_sha256
    || item.content_sha256 || item.reviewState?.content_sha256;
  if (!revision) return url;
  const [path, fragment] = url.split("#", 2);
  return `${path}${path.includes("?") ? "&" : "?"}v=${encodeURIComponent(revision)}${fragment === undefined ? "" : `#${fragment}`}`;
}

function isSafeLocalUrl(raw) {
  if (!String(raw).startsWith("/") && !/^https?:\/\//i.test(String(raw))) return false;
  try {
    const url = new URL(String(raw), window.location.origin);
    return url.origin === window.location.origin;
  } catch (_) {
    return false;
  }
}

function seekAudio(seconds) {
  if (!dom.detailAudio.src) {
    showToast("このデータには再生できる音声がありません。", true);
    return;
  }
  dom.audioSection.open = true;
  dom.detailAudio.currentTime = Math.max(0, seconds);
  const play = dom.detailAudio.play();
  if (play && typeof play.catch === "function") play.catch(() => {});
}

function highlightCurrentAlignment() {
  const current = dom.detailAudio.currentTime;
  let active = null;
  dom.alignmentBody.querySelectorAll(".alignment-row").forEach((row) => {
    const start = Number(row.dataset.start);
    const end = Number(row.dataset.end);
    const match = current >= start && current < Math.max(end, start + 0.08);
    row.classList.toggle("is-active", match);
    if (match) active = row;
  });
  if (active && !active.dataset.wasVisible) {
    active.dataset.wasVisible = "true";
    active.scrollIntoView({ block: "nearest" });
  }
}

function clearActiveAlignment() {
  dom.alignmentBody.querySelectorAll(".alignment-row").forEach((row) => {
    row.classList.remove("is-active");
    delete row.dataset.wasVisible;
  });
}

function navigateDetail(direction) {
  const token = state.navigation[direction];
  if (token) void selectItem(token);
}

function navigationToken(value) {
  if (!value) return "";
  if (typeof value === "string" || typeof value === "number") return String(value);
  return String(value.token ?? value.item_token ?? value.id ?? value.uid ?? "");
}

function listNeighbor(token, offset) {
  const index = state.items.findIndex((item) => item.token === token);
  if (index < 0) return "";
  return state.items[index + offset]?.token || "";
}

function updateSelectedRow() {
  dom.datasetList.querySelectorAll(".dataset-item").forEach((row) => {
    const selected = row.dataset.token === state.selectedToken;
    row.classList.toggle("is-selected", selected);
    row.setAttribute("aria-current", selected ? "true" : "false");
  });
}

function clearFilters(reload = true) {
  state.filters = {
    collection: "", category: "", directory: "", stage: "", review_status: "",
    auxiliary_kind: "", review_required: "", ai_status: "", q: "",
  };
  state.page = 1;
  dom.categoryFilter.value = "";
  dom.stageFilter.value = "";
  dom.reviewStatusFilter.value = "";
  dom.auxiliaryKindFilter.value = "";
  dom.aiStatusFilter.value = "";
  dom.reviewRequiredFilter.checked = false;
  dom.uidSearch.value = "";
  dom.annotationCollection.value = "";
  dom.activeCollection.textContent = "ALL DATA";
  updateTreeSelection();
  if (reload !== false) void loadItems({ selectFirst: true });
}

function filterParams() {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) params.set(key, value);
  }
  if (state.annotationMode) params.set("has_dialogue", "true");
  return params;
}

function setViewMode(annotation, { reload = true, updateUrl = true } = {}) {
  const changed = state.annotationMode !== annotation;
  state.annotationMode = annotation;
  if (changed) dom.reviewContextDetails.open = !annotation;
  // Entering the review workspace starts with the outstanding mandatory queue.
  // Existing selections, filters and editor DOM are otherwise left untouched.
  if (annotation && !state.selectedToken && !Object.values(state.filters).some(Boolean)) {
    state.filters.review_status = "pending";
    state.filters.review_required = "true";
    dom.reviewStatusFilter.value = "pending";
    dom.reviewRequiredFilter.checked = true;
  }
  if (updateUrl) window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}${annotation ? "#annotate" : ""}`);
  updateAnnotationLayout();
  updateBackdrop();
  scheduleDialogueEditorSizing();
  if (reload && !state.editingDialogue && !state.savingDialogue && !state.reviewBusy) void loadItems();
}

function updateAnnotationLayout() {
  document.body.classList.toggle("annotation-mode", state.annotationMode);
  document.body.classList.toggle("annotation-list-hidden", state.annotationMode && state.annotationListHidden);
  document.body.classList.toggle("annotation-advanced-open", state.annotationMode && state.annotationAdvanced);
  dom.annotationMode.classList.toggle("is-active", state.annotationMode);
  dom.browseMode.classList.toggle("is-active", !state.annotationMode);
  dom.annotationMode.setAttribute("aria-pressed", String(state.annotationMode));
  dom.browseMode.setAttribute("aria-pressed", String(!state.annotationMode));
  dom.annotationListToggle.textContent = state.annotationListHidden ? "一覧を表示" : "一覧を隠す";
  dom.annotationListToggle.setAttribute("aria-expanded", String(!state.annotationListHidden));
  dom.annotationAdvancedToggle.setAttribute("aria-expanded", String(state.annotationAdvanced));
  dom.annotationAdvancedToggle.textContent = state.annotationAdvanced ? "詳細条件を閉じる" : "詳細条件";
}

function updateCategoryOptions(source) {
  let categories = source;
  if (!Array.isArray(categories) && categories && typeof categories === "object") categories = Object.keys(categories);
  if (!Array.isArray(categories)) return;
  const current = state.filters.category || dom.categoryFilter.value;
  const existing = new Set(Array.from(dom.categoryFilter.options).map((option) => option.value));
  for (const entry of categories) {
    const value = String(typeof entry === "string" ? entry : (entry.value ?? entry.category ?? entry.name ?? entry.label ?? ""));
    if (!value || existing.has(value)) continue;
    const label = String(typeof entry === "string" ? entry : (entry.label ?? entry.name ?? entry.category ?? value));
    const count = typeof entry === "object" ? asNumber(entry.count ?? entry.total) : null;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = count === null ? label : `${label}（${formatNumber(count)}）`;
    dom.categoryFilter.append(option);
    existing.add(value);
  }
  dom.categoryFilter.value = current;
}

function openTree() {
  dom.treePanel.classList.add("is-open");
  updateBackdrop();
}

function closeTree() {
  dom.treePanel.classList.remove("is-open");
  updateBackdrop();
}

function openDetail() {
  dom.detailPanel.classList.add("is-open");
  updateBackdrop();
}

function closeDetail() {
  if (state.savingDialogue) return;
  if (state.editingDialogue) {
    if (!window.confirm("未保存の編集を取り消しますか？")) return;
    cancelDialogueEdit();
  }
  dom.detailPanel.classList.remove("is-open");
  updateBackdrop();
}

function closeOverlays() {
  dom.treePanel.classList.remove("is-open");
  closeDetail();
}

function updateBackdrop() {
  if (state.annotationMode) { dom.mobileBackdrop.hidden = true; return; }
  const treeOverlay = window.matchMedia("(max-width: 980px)").matches && dom.treePanel.classList.contains("is-open");
  const detailOverlay = window.matchMedia("(max-width: 1390px)").matches && dom.detailPanel.classList.contains("is-open");
  dom.mobileBackdrop.hidden = !(treeOverlay || detailOverlay);
}

function showToast(message, error = false) {
  window.clearTimeout(state.toastTimer);
  dom.toast.textContent = message;
  dom.toast.classList.toggle("is-error", error);
  dom.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, error ? 6500 : 4000);
}

function createElement(tag, className = "", text = undefined) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function readCount(source, keys) {
  for (const key of keys) {
    const value = source?.[key];
    const parsed = asNumber(value && typeof value === "object" ? (value.count ?? value.total ?? value.value) : value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function asNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function integerValue(value) {
  const number = asNumber(value);
  return number !== null && Number.isInteger(number) && number >= 0 ? number : null;
}

function booleanValue(value) {
  if (value === true || value === false) return value;
  if (value === 1 || value === "1") return true;
  if (value === 0 || value === "0") return false;
  const text = String(value ?? "").trim().toLowerCase();
  if (["true", "yes", "required"].includes(text)) return true;
  if (["false", "no", "not_required", "not-required"].includes(text)) return false;
  return null;
}

function secondsValue(value) {
  if (Array.isArray(value)) return secondsValue(value[0]);
  if (typeof value === "string" && value.includes(":")) {
    const parts = value.split(":").map(Number);
    if (parts.every(Number.isFinite)) return parts.reduce((total, part) => total * 60 + part, 0);
  }
  const number = asNumber(value);
  if (number === null) return 0;
  // Some alignment files store integer milliseconds.
  return number > 10000 ? number / 1000 : number;
}

function formatNumber(value) {
  return new Intl.NumberFormat("ja-JP").format(value);
}

function formatAudioDuration(value) {
  const seconds = asNumber(value);
  if (seconds === null || seconds < 0) return "—";
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remainder = rounded % 60;
  if (hours > 0) return `${formatNumber(hours)}時間${minutes}分${remainder}秒`;
  if (minutes > 0) return `${minutes}分${remainder}秒`;
  return `${remainder}秒`;
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("ja-JP", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

function toCamel(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}
