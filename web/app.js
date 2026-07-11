const state = {
  browseLogId: "",
  payload: null,
  plotPayload: null,
  plotLoadToken: 0,
  plotTrackers: {},
  sharedPlotTracker: null,
  pendingPlotTracker: null,
  plotTrackerFrame: null,
  plotScrollFrame: null,
  plotResizeTimer: null,
  plotRenderMetrics: {},
  hiddenPlotSeries: {},
  analysisRunId: null,
  analysisPollTimer: null,
  sidebarCollapsed: false,
  plotSidebarCollapsed: false,
  availableTags: [],
  currentLogTags: [],
  tagStatus: "",
};

const els = {
  appShell: document.getElementById("appShell"),
  runStatus: document.getElementById("runStatus"),
  downloadLog: document.getElementById("downloadLog"),
  downloadParameters: document.getElementById("downloadParameters"),
  downloadNonDefaultParameters: document.getElementById("downloadNonDefaultParameters"),
  downloadKml: document.getElementById("downloadKml"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebarToggle"),
  sidebarSummary: document.getElementById("sidebarSummary"),
  plotSidebar: document.getElementById("plotSidebar"),
  plotSidebarToggle: document.getElementById("plotSidebarToggle"),
  plotSidebarStatus: document.getElementById("plotSidebarStatus"),
  plotNavigationMenu: document.getElementById("plotNavigationMenu"),
  plotNavigation: document.getElementById("plotNavigation"),
  plotRows: document.getElementById("plotRows"),
  factGrid: document.getElementById("factGrid"),
  warningsList: document.getElementById("warningsList"),
  warningCount: document.getElementById("warningCount"),
  messageSearch: document.getElementById("messageSearch"),
  messageFilter: document.getElementById("messageFilter"),
  parameterSearch: document.getElementById("parameterSearch"),
  parameterStatus: document.getElementById("parameterStatus"),
  hideRcCal: document.getElementById("hideRcCal"),
  parameterRows: document.getElementById("parameterRows"),
  parameterCount: document.getElementById("parameterCount"),
  topicSearch: document.getElementById("topicSearch"),
  topicRows: document.getElementById("topicRows"),
  topicCount: document.getElementById("topicCount"),
  mainSummary: document.getElementById("mainSummary"),
  analysisSourcePath: document.getElementById("analysisSourcePath"),
  analysisQuestion: document.getElementById("analysisQuestion"),
  analysisButton: document.getElementById("analysisButton"),
  analysisStatus: document.getElementById("analysisStatus"),
  analysisProgress: document.getElementById("analysisProgress"),
  analysisReport: document.getElementById("analysisReport"),
  changedParameterRows: document.getElementById("changedParameterRows"),
  timelineRows: document.getElementById("timelineRows"),
  logTagPanel: document.getElementById("logTagPanel"),
  logTagList: document.getElementById("logTagList"),
  logTagStatus: document.getElementById("logTagStatus"),
  logTagSelect: document.getElementById("logTagSelect"),
  logTagAdd: document.getElementById("logTagAdd"),
  logTagNewName: document.getElementById("logTagNewName"),
  logTagCreate: document.getElementById("logTagCreate"),
};

els.parameterSearch.addEventListener("input", renderParameters);
els.parameterStatus.addEventListener("change", renderParameters);
els.hideRcCal.addEventListener("change", renderParameters);
els.messageSearch.addEventListener("input", renderWarnings);
els.messageFilter.addEventListener("change", renderWarnings);
els.topicSearch.addEventListener("input", renderTopics);
els.analysisButton.addEventListener("click", startAnalysis);
els.logTagAdd.addEventListener("click", addSelectedLogTag);
els.logTagCreate.addEventListener("click", createAndAddLogTag);
els.logTagNewName.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    createAndAddLogTag();
  }
});
els.logTagList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove-log-tag]");
  if (!button) return;
  removeCurrentLogTag(button.dataset.removeLogTag);
});
els.plotNavigation.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-plot-target]");
  if (!link) return;
  event.preventDefault();
  const plot = document.getElementById(link.dataset.plotTarget);
  if (plot) plot.scrollIntoView({ block: "start" });
  els.plotNavigationMenu.removeAttribute("open");
});
els.plotSidebar.addEventListener("scroll", scheduleVisiblePlotTrackerDraw, { passive: true });
document.addEventListener("click", (event) => {
  if (!els.plotNavigationMenu.open || els.plotNavigationMenu.contains(event.target)) return;
  els.plotNavigationMenu.removeAttribute("open");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") els.plotNavigationMenu.removeAttribute("open");
});
const initialParams = new URLSearchParams(window.location.search);
const initialBrowseId = initialParams.get("browse_id");
if (initialBrowseId) {
  state.browseLogId = initialBrowseId;
  configureDownloadLinks(initialBrowseId);
  window.requestAnimationFrame(() => loadReview());
} else {
  setStatus("Missing log");
  els.mainSummary.innerHTML = `<p class="message-item">Choose a log from <a href="/browse">Browse</a> or <a href="/upload">Upload</a>.</p>`;
}

els.sidebarToggle.addEventListener("click", () => {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  els.appShell.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  els.sidebar.classList.toggle("collapsed", state.sidebarCollapsed);
  els.sidebarToggle.title = state.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar";
  els.sidebarToggle.setAttribute(
    "aria-label",
    state.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar",
  );
  els.sidebarToggle.querySelector("span").textContent = state.sidebarCollapsed ? "›" : "‹";
});

els.plotSidebarToggle.addEventListener("click", () => {
  state.plotSidebarCollapsed = !state.plotSidebarCollapsed;
  els.appShell.classList.toggle("plot-sidebar-collapsed", state.plotSidebarCollapsed);
  els.plotSidebar.classList.toggle("collapsed", state.plotSidebarCollapsed);
  els.plotSidebarToggle.title = state.plotSidebarCollapsed ? "Expand plots" : "Collapse plots";
  els.plotSidebarToggle.setAttribute(
    "aria-label",
    state.plotSidebarCollapsed ? "Expand plots" : "Collapse plots",
  );
  els.plotSidebarToggle.querySelector("span").textContent = state.plotSidebarCollapsed ? "‹" : "›";
  els.plotNavigationMenu.removeAttribute("open");
  if (!state.plotSidebarCollapsed) schedulePlotResize();
});

window.addEventListener("resize", () => {
  schedulePlotResize();
});

async function loadReview() {
  setStatus("Loading...");
  try {
    const result = await fetchJson("/api/preparse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ browse_log_id: state.browseLogId }),
    });
    state.payload = result;
    state.plotPayload = null;
    if (state.plotTrackerFrame != null) {
      window.cancelAnimationFrame(state.plotTrackerFrame);
      state.plotTrackerFrame = null;
    }
    if (state.plotScrollFrame != null) {
      window.cancelAnimationFrame(state.plotScrollFrame);
      state.plotScrollFrame = null;
    }
    state.plotTrackers = {};
    state.sharedPlotTracker = null;
    state.pendingPlotTracker = null;
    state.plotRenderMetrics = {};
    state.hiddenPlotSeries = {};
    state.availableTags = [];
    state.currentLogTags = result?.browse?.row?.tags || [];
    state.tagStatus = "";
    if (result?.browse?.log_id) {
      state.browseLogId = result.browse.log_id;
    }
    els.analysisSourcePath.value = result?.inputs?.source_path || "";
    updateKmlDownloadAvailability();
    await refreshLoadedLogTags();
    resetAnalysis();
    renderAll();
    loadInteractivePlots();
    setStatus("Loaded");
  } catch (error) {
    setStatus("Failed");
    els.mainSummary.innerHTML = `<p class="message-item">${escapeHtml(error.message)}</p>`;
  }
}

async function startAnalysis() {
  if (!state.payload) {
    setAnalysisStatus("Load a log first");
    return;
  }

  const inputs = state.payload.inputs || {};
  const currentSourcePath = stringOrNull(els.analysisSourcePath.value);
  const question = els.analysisQuestion.value.trim()
    || "Analyze this flight log and identify the most likely root causes.";
  const payload = {
    log_path: inputs.log_path,
    mission_path: inputs.mission_path,
    source_path: currentSourcePath || inputs.source_path,
    user_question: question,
  };

  if (!payload.log_path) {
    setAnalysisStatus("Missing log path");
    return;
  }

  clearAnalysisPoll();
  els.analysisButton.disabled = true;
  els.analysisReport.innerHTML = "";
  els.analysisProgress.innerHTML = `<div class="progress-item active">Starting analysis...</div>`;
  setAnalysisStatus("Starting");

  try {
    const response = await fetch("/api/analyze-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    state.analysisRunId = result.run_id;
    renderAnalysisRun(result);
    state.analysisPollTimer = window.setInterval(pollAnalysisRun, 1000);
  } catch (error) {
    setAnalysisStatus("Failed");
    els.analysisProgress.innerHTML = `<div class="progress-item failed">${escapeHtml(error.message)}</div>`;
    els.analysisButton.disabled = false;
  }
}

async function pollAnalysisRun() {
  if (!state.analysisRunId) return;

  try {
    const response = await fetch(`/api/analyze-runs/${encodeURIComponent(state.analysisRunId)}`);
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    renderAnalysisRun(result);
    if (result.status === "completed" || result.status === "failed") {
      clearAnalysisPoll();
      els.analysisButton.disabled = !state.payload;
    }
  } catch (error) {
    clearAnalysisPoll();
    setAnalysisStatus("Failed");
    els.analysisProgress.innerHTML += `<div class="progress-item failed">${escapeHtml(error.message)}</div>`;
    els.analysisButton.disabled = !state.payload;
  }
}

function renderAnalysisRun(run) {
  setAnalysisStatus(statusText(run.status, run.progress?.phase));
  const messages = run.progress?.messages || [];
  els.analysisProgress.innerHTML = messages.length
    ? messages.map((message, index) => renderProgressItem(message, index === messages.length - 1, run.status)).join("")
    : `<div class="progress-item active">${escapeHtml(statusText(run.status, ""))}</div>`;

  if (run.status === "failed") {
    els.analysisReport.innerHTML = `<p class="message-item">${escapeHtml(run.error || "Analysis failed.")}</p>`;
  } else if (run.report) {
    renderAnalysisReport(run.report);
  }
}

function renderProgressItem(item, isLatest, status) {
  const classes = ["progress-item"];
  if (isLatest && status !== "completed") classes.push(status === "failed" ? "failed" : "active");
  if (item.event === "run.finished") classes.push("done");
  const time = item.ts ? `<span>${escapeHtml(formatEventTime(item.ts))}</span>` : "";
  return `
    <div class="${classes.join(" ")}">
      <strong>${escapeHtml(item.message)}</strong>
      ${time}
    </div>
  `;
}

function renderAnalysisReport(report) {
  const hypotheses = report.ranked_hypotheses || [];
  const intentSummary = report.question_intent_summary || "";
  const legacyAssumptions = report.assumption_header || "";
  const legacyTimeline = report.timeline_summary || "";
  els.analysisReport.innerHTML = `
    <div class="report-block">
      <h3>Summary</h3>
      <p>${escapeHtml(report.final_summary || "")}</p>
    </div>
    ${intentSummary ? renderReportBlock("Analysis Intent", intentSummary) : ""}
    ${legacyAssumptions ? renderReportBlock("Assumptions", legacyAssumptions) : ""}
    ${legacyTimeline ? renderReportBlock("Timeline", legacyTimeline) : ""}
    ${renderStringList("Confirmed", report.confirmed || [])}
    ${renderStringList("Unconfirmed", report.unconfirmed || [])}
    ${renderStringList("Excluded Mechanisms", report.excluded_mechanisms || [])}
    <div class="hypothesis-list">
      ${hypotheses.map(renderHypothesis).join("")}
    </div>
  `;
}

function renderReportBlock(title, text) {
  return `
    <div class="report-block">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
}

function renderHypothesis(hypothesis, index) {
  const applicability = hypothesis.applicability || {};
  return `
    <article class="hypothesis-card">
      <div class="section-title-row">
        <h3>${escapeHtml(`${index + 1}. ${hypothesis.title || "Hypothesis"}`)}</h3>
        <span class="status-badge">${escapeHtml(hypothesis.confidence || "unknown")}</span>
      </div>
      ${hypothesis.known_px4_mechanism ? `<p><strong>PX4 mechanism:</strong> ${escapeHtml(hypothesis.known_px4_mechanism)}</p>` : ""}
      <p>${escapeHtml(hypothesis.mechanism || "")}</p>
      ${renderStringList("Evidence", hypothesis.evidence || [])}
      ${renderStringList("Contradicting Evidence", hypothesis.contradicting_evidence || [])}
      ${renderStringList("Supported Conditions", applicability.supported_conditions || [])}
      ${renderStringList("Unresolved Conditions", applicability.unresolved_conditions || [])}
      ${renderStringList("Excluded By", applicability.excluded_by || [])}
      ${renderStringList("Relevant Parameters", formatParameterValues(applicability.relevant_parameters || []))}
      ${renderStringList("Missing Signals", applicability.missing_required_signals || [])}
      ${renderSourceRefs(hypothesis.source_refs || hypothesis.code_references || [])}
      ${renderSignatureItems(hypothesis.expected_logged_signature || [])}
      ${renderCheckList("Numeric Checks", hypothesis.numeric_checks || [])}
      ${renderCheckList("Exclusion Checks", hypothesis.exclusion_checks || [])}
      ${renderPlots(hypothesis.plots || [])}
    </article>
  `;
}

function renderStringList(title, rows) {
  if (!rows.length) return "";
  return `
    <div class="report-list">
      <h4>${escapeHtml(title)}</h4>
      <ul>
        ${rows.map((row) => `<li>${escapeHtml(formatReportValue(row))}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderSourceRefs(refs) {
  if (!refs.length) return "";
  return `
    <div class="report-list">
      <h4>Source References</h4>
      <ul>
        ${refs.map((ref) => `
          <li>
            <code>${escapeHtml(sourceRefLabel(ref))}</code>
            ${ref.explanation ? ` - ${escapeHtml(ref.explanation)}` : ""}
          </li>
        `).join("")}
      </ul>
    </div>
  `;
}

function renderSignatureItems(items) {
  if (!items.length) return "";
  return `
    <div class="report-list">
      <h4>Expected Logged Signature</h4>
      <ul>
        ${items.map((item) => `
          <li>
            <strong>${escapeHtml(item.name || "signature")}</strong>
            ${item.signal ? ` <code>${escapeHtml(item.signal)}</code>` : ""}
            ${item.description ? ` - ${escapeHtml(item.description)}` : ""}
            ${item.expected_behavior ? ` (${escapeHtml(item.expected_behavior)})` : ""}
          </li>
        `).join("")}
      </ul>
    </div>
  `;
}

function renderCheckList(title, checks) {
  if (!checks.length) return "";
  return `
    <div class="report-list">
      <h4>${escapeHtml(title)}</h4>
      <ul>
        ${checks.map((check) => `
          <li>
            <code>${escapeHtml(check.type || "check")}</code>
            ${check.window ? ` in ${escapeHtml(check.window)}` : ""}
            ${check.signal ? ` on <code>${escapeHtml(check.signal)}</code>` : ""}
            ${check.actual || check.setpoint ? ` ${escapeHtml([check.actual, check.setpoint].filter(Boolean).join(" vs "))}` : ""}
            ${check.supports ? ` - ${escapeHtml(check.supports)}` : ""}
            ${check.contradicts ? ` - ${escapeHtml(check.contradicts)}` : ""}
            ${check.description && !check.supports && !check.contradicts ? ` - ${escapeHtml(check.description)}` : ""}
          </li>
        `).join("")}
      </ul>
    </div>
  `;
}

function sourceRefLabel(ref) {
  const lineRange = ref.start_line
    ? `:${ref.start_line}${ref.end_line ? `-${ref.end_line}` : ""}`
    : "";
  const fn = ref.function ? ` ${ref.function}` : "";
  return `${ref.file || "source"}${lineRange}${fn}`;
}

function formatReportValue(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function formatParameterValues(parameters) {
  return parameters.map((parameter) => {
    if (!parameter || typeof parameter !== "object") return formatReportValue(parameter);
    return `${parameter.name || "parameter"}=${parameter.value ?? ""}`;
  });
}

function renderPlots(plots) {
  if (!plots.length) return "";
  return `
    <div class="plot-grid">
      ${plots.map((plot) => `
        <figure class="plot-card">
          ${plot.path ? `<img src="${escapeAttr(artifactUrl(plot.path))}" alt="${escapeAttr(plot.title || "Analysis plot")}">` : ""}
          <figcaption>
            <strong>${escapeHtml(plot.title || "Plot")}</strong>
            <span>${escapeHtml(plot.purpose || "")}</span>
            ${plot.warnings?.length ? `<span class="plot-warning">${escapeHtml(plot.warnings.join("; "))}</span>` : ""}
          </figcaption>
        </figure>
      `).join("")}
    </div>
  `;
}

function resetAnalysis() {
  clearAnalysisPoll();
  state.analysisRunId = null;
  els.analysisButton.disabled = !state.payload;
  els.analysisStatus.textContent = "Idle";
  els.analysisProgress.innerHTML = "";
  els.analysisReport.innerHTML = "";
}

function clearAnalysisPoll() {
  if (state.analysisPollTimer) {
    window.clearInterval(state.analysisPollTimer);
    state.analysisPollTimer = null;
  }
}

async function refreshLoadedLogTags() {
  const logId = currentBrowseLogId();
  if (!logId) {
    state.availableTags = [];
    state.currentLogTags = [];
    state.tagStatus = "Not indexed";
    return;
  }

  try {
    const [tagsPayload, logPayload] = await Promise.all([
      fetchJson("/api/browse-tags"),
      fetchJson(`/api/browse-log?log_id=${encodeURIComponent(logId)}`),
    ]);
    state.availableTags = tagsPayload.tags || [];
    state.currentLogTags = logPayload.log?.tags || [];
    state.tagStatus = "";
    if (state.payload?.browse) {
      state.payload.browse.row = logPayload.log;
      state.payload.browse.log_id = logPayload.log?.id || state.payload.browse.log_id;
    }
  } catch (error) {
    state.tagStatus = error.message;
  }
}

async function addSelectedLogTag() {
  const tag = els.logTagSelect.value;
  if (!tag) return;
  await updateCurrentLogTag(tag, "add");
}

async function createAndAddLogTag() {
  const name = els.logTagNewName.value.trim();
  if (!name) return;
  const logId = currentBrowseLogId();
  if (!logId) return;

  els.logTagCreate.disabled = true;
  try {
    await fetchJson("/api/browse-tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await updateCurrentLogTag(name, "add");
    els.logTagNewName.value = "";
  } catch (error) {
    state.tagStatus = error.message;
    renderLogTags();
  } finally {
    els.logTagCreate.disabled = false;
  }
}

async function removeCurrentLogTag(tag) {
  if (!tag) return;
  await updateCurrentLogTag(tag, "remove");
}

async function updateCurrentLogTag(tag, action) {
  const logId = currentBrowseLogId();
  if (!logId) return;

  els.logTagAdd.disabled = true;
  try {
    await fetchJson("/api/browse-log-tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_id: logId, tag, action }),
    });
    await refreshLoadedLogTags();
    renderLogTags();
  } catch (error) {
    state.tagStatus = error.message;
    renderLogTags();
  } finally {
    renderLogTags();
  }
}

function currentBrowseLogId() {
  return state.payload?.browse?.log_id || state.payload?.browse?.row?.id || state.browseLogId;
}

function configureDownloadLinks(logId) {
  const baseUrl = `/api/browse-download?log_id=${encodeURIComponent(logId)}&type=`;
  els.downloadLog.href = `${baseUrl}ulog`;
  els.downloadParameters.href = `${baseUrl}parameters`;
  els.downloadNonDefaultParameters.href = `${baseUrl}parameters_non_default`;
  els.downloadKml.href = `${baseUrl}kml`;
}

function updateKmlDownloadAvailability() {
  const hasGpsPosition = (state.payload?.topics || []).some(
    (topic) => topic.name === "vehicle_gps_position" && !topic.missing_expected,
  );
  els.downloadKml.hidden = !hasGpsPosition;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}

function renderAll() {
  renderFacts();
  renderWarnings();
  renderParameters();
  renderTopics();
  renderMainSummary();
  renderLogTags();
  renderChangedParameters();
  renderTimeline();
  renderInteractivePlots();
}

async function loadInteractivePlots() {
  const logPath = state.payload?.inputs?.log_path;
  if (!logPath) {
    els.plotSidebarStatus.textContent = "Missing log path";
    els.plotRows.innerHTML = `<p class="plot-empty">No log path available for plots.</p>`;
    return;
  }

  const token = state.plotLoadToken + 1;
  state.plotLoadToken = token;
  els.plotSidebarStatus.textContent = "Loading plots...";
  els.plotRows.innerHTML = `<p class="plot-empty">Extracting plot samples...</p>`;

  try {
    const response = await fetch("/api/interactive-plots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ log_path: logPath }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    if (token !== state.plotLoadToken) return;

    state.plotPayload = result;
    initializePlotTrackers(result.plots || []);
    renderInteractivePlots();
  } catch (error) {
    if (token !== state.plotLoadToken) return;
    els.plotSidebarStatus.textContent = "Plot load failed";
    els.plotRows.innerHTML = `<p class="plot-empty">${escapeHtml(error.message)}</p>`;
  }
}

function initializePlotTrackers(plots) {
  if (state.sharedPlotTracker == null) {
    const [payloadStart, payloadEnd] = state.plotPayload?.time_range_s || [];
    if (Number.isFinite(payloadStart) && Number.isFinite(payloadEnd) && payloadEnd > payloadStart) {
      state.sharedPlotTracker = payloadStart + (payloadEnd - payloadStart) * 0.5;
    }
  }

  plots.forEach((plot) => {
    const [start, end] = plot.time_range_s || [0, 0];
    if (state.plotTrackers[plot.id] !== undefined) return;
    const initial = Number.isFinite(state.sharedPlotTracker)
      ? state.sharedPlotTracker
      : Number.isFinite(start) ? start : 0;
    state.plotTrackers[plot.id] = clampTime(initial, start, end);
  });
}

function renderInteractivePlots() {
  const plots = state.plotPayload?.plots || [];
  if (!state.payload) {
    els.plotSidebarStatus.textContent = "Load a log";
    renderPlotNavigation([]);
    els.plotRows.innerHTML = `<p class="plot-empty">Load a ULog to view Flight Review plots.</p>`;
    return;
  }

  if (!state.plotPayload) {
    els.plotSidebarStatus.textContent = "Plots pending";
    renderPlotNavigation([]);
    els.plotRows.innerHTML = `<p class="plot-empty">Plots will load after pre-parse completes.</p>`;
    return;
  }

  els.plotSidebarStatus.textContent = plots.length ? `${plots.length} plots` : "No plottable data";
  renderPlotNavigation(plots);
  els.plotRows.innerHTML = plots.length
    ? plots.map((plot, index) => renderInteractivePlot(plot, index)).join("")
    : `<p class="plot-empty">No Flight Review plot signals were available in this log.</p>`;

  bindInteractivePlots(plots);
  window.requestAnimationFrame(() => drawInteractivePlots(plots));
}

function renderPlotNavigation(plots) {
  els.plotNavigationMenu.hidden = !plots.length;
  if (!plots.length) els.plotNavigationMenu.removeAttribute("open");
  els.plotNavigation.innerHTML = plots.map((plot, index) => `
    <a href="#plot-${index}" data-plot-target="plot-${index}">${escapeHtml(plot.title || `Plot ${index + 1}`)}</a>
  `).join("");
}

function renderInteractivePlot(plot, index) {
  const [start, end] = plot.time_range_s || [0, 0];
  const tracker = clampTracker(plot.id, start, end);
  const sources = plot.kind === "local_position"
    ? (plot.traces || [])
    : plot.kind === "spectrogram" ? [] : (plot.series || []);
  const step = Math.max((end - start) / 1000, 0.001);

  return `
    <section class="interactive-plot" id="plot-${index}" data-plot-id="${escapeAttr(plot.id)}">
      <div class="interactive-plot-header">
        <h3>${escapeHtml(plot.title)}</h3>
        <span class="plot-time" id="plotTime-${escapeAttr(plot.id)}">${escapeHtml(formatLogTime(tracker))}</span>
      </div>
      <div class="plot-canvas-stack">
        <canvas class="plot-canvas" id="plotCanvas-${escapeAttr(plot.id)}"></canvas>
        <canvas class="plot-tracker-canvas" id="plotTrackerCanvas-${escapeAttr(plot.id)}" aria-hidden="true"></canvas>
      </div>
      <div class="plot-tracker">
        <input
          id="plotTracker-${escapeAttr(plot.id)}"
          type="range"
          min="${escapeAttr(start)}"
          max="${escapeAttr(end)}"
          step="${escapeAttr(step)}"
          value="${escapeAttr(tracker)}"
          ${end <= start ? "disabled" : ""}
        >
      </div>
      <div class="plot-legend">
        ${sources.map((source) => renderPlotToggle(plot.id, source)).join("")}
      </div>
      <div class="plot-readout" id="plotReadout-${escapeAttr(plot.id)}"></div>
    </section>
  `;
}

function renderPlotToggle(plotId, source) {
  const checked = isPlotSourceVisible(plotId, source.key) ? "checked" : "";
  return `
    <label class="plot-toggle">
      <input type="checkbox" data-plot-id="${escapeAttr(plotId)}" data-source-key="${escapeAttr(source.key)}" ${checked}>
      <span class="legend-swatch" style="background:${escapeAttr(source.color)}"></span>
      ${escapeHtml(source.label)}
    </label>
  `;
}

function bindInteractivePlots(plots) {
  plots.forEach((plot) => {
    const slider = document.getElementById(`plotTracker-${plot.id}`);
    const canvas = document.getElementById(`plotCanvas-${plot.id}`);
    if (slider) {
      slider.addEventListener("input", () => {
        setSharedPlotTracker(Number(slider.value));
      });
    }

    if (canvas && plot.kind !== "local_position") {
      canvas.addEventListener("click", (event) => {
        const time = canvasTimeFromEvent(canvas, plot, event);
        if (time == null) return;
        setSharedPlotTracker(time);
      });
    }
  });

  els.plotRows.querySelectorAll(".plot-toggle input").forEach((input) => {
    input.addEventListener("change", () => {
      const plotId = input.dataset.plotId;
      const sourceKey = input.dataset.sourceKey;
      if (!plotId || !sourceKey) return;
      state.hiddenPlotSeries[`${plotId}:${sourceKey}`] = !input.checked;
      const plot = (state.plotPayload?.plots || []).find((candidate) => candidate.id === plotId);
      if (plot) drawInteractivePlot(plot);
    });
  });
}

function setSharedPlotTracker(time) {
  if (!Number.isFinite(time)) return;
  state.pendingPlotTracker = time;
  if (state.plotTrackerFrame != null) return;
  state.plotTrackerFrame = window.requestAnimationFrame(flushSharedPlotTracker);
}

function flushSharedPlotTracker() {
  state.plotTrackerFrame = null;
  const time = state.pendingPlotTracker;
  state.pendingPlotTracker = null;
  if (!Number.isFinite(time)) return;

  const plots = state.plotPayload?.plots || [];
  state.sharedPlotTracker = time;
  plots.forEach((plot) => {
    const [start, end] = plot.time_range_s || [0, 0];
    const tracker = clampTime(time, start, end);
    state.plotTrackers[plot.id] = tracker;
    const slider = document.getElementById(`plotTracker-${plot.id}`);
    if (slider) slider.value = String(tracker);
  });
  drawInteractivePlotTrackers(plots, { visibleOnly: true });
}

function scheduleVisiblePlotTrackerDraw() {
  if (state.plotScrollFrame != null) return;
  state.plotScrollFrame = window.requestAnimationFrame(() => {
    state.plotScrollFrame = null;
    drawInteractivePlotTrackers(state.plotPayload?.plots || [], { visibleOnly: true });
  });
}

function schedulePlotResize() {
  if (state.plotResizeTimer != null) {
    window.clearTimeout(state.plotResizeTimer);
  }
  state.plotResizeTimer = window.setTimeout(() => {
    state.plotResizeTimer = null;
    drawResizedInteractivePlots(state.plotPayload?.plots || []);
  }, 100);
}

function drawInteractivePlots(plots) {
  plots.forEach(drawInteractivePlot);
}

function drawInteractivePlotTrackers(plots, { visibleOnly = false } = {}) {
  plots.forEach((plot) => {
    if (!visibleOnly || isPlotNearViewport(plot)) drawInteractivePlotTracker(plot);
  });
}

function isPlotNearViewport(plot) {
  const canvas = document.getElementById(`plotTrackerCanvas-${plot.id}`);
  const section = canvas?.closest(".interactive-plot");
  if (!section) return false;
  const plotRect = section.getBoundingClientRect();
  const sidebarRect = els.plotSidebar.getBoundingClientRect();
  const margin = 200;
  return plotRect.bottom >= sidebarRect.top - margin
    && plotRect.top <= sidebarRect.bottom + margin;
}

function drawResizedInteractivePlots(plots) {
  plots.forEach((plot) => {
    const canvas = document.getElementById(`plotCanvas-${plot.id}`);
    if (canvas && canvasNeedsResize(canvas)) drawInteractivePlot(plot);
  });
}

function drawInteractivePlot(plot) {
  const canvas = document.getElementById(`plotCanvas-${plot.id}`);
  if (!canvas) return;

  const ctx = prepareCanvas(canvas);
  const bounds = plotBounds(canvas);
  drawPlotFrame(ctx, canvas, bounds);

  if (plot.kind === "local_position") {
    drawLocalPositionPlot(ctx, canvas, bounds, plot);
  } else if (plot.kind === "spectrogram") {
    drawSpectrogramPlot(ctx, canvas, bounds, plot);
  } else {
    drawTimeseriesPlot(ctx, canvas, bounds, plot);
  }

  drawInteractivePlotTracker(plot);
}

function drawInteractivePlotTracker(plot) {
  const canvas = document.getElementById(`plotTrackerCanvas-${plot.id}`);
  if (!canvas) return;

  const ctx = prepareCanvas(canvas);
  const bounds = plotBounds(canvas);
  ctx.clearRect(0, 0, bounds.width, bounds.height);

  if (plot.kind === "local_position") {
    drawLocalPositionTracker(ctx, bounds, plot);
  } else if (plot.kind === "spectrogram") {
    drawSpectrogramTracker(ctx, bounds, plot);
  } else {
    drawTimeseriesTracker(ctx, bounds, plot);
  }

  updatePlotReadout(plot);
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function canvasNeedsResize(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return false;
  return canvas.width !== Math.round(rect.width * ratio)
    || canvas.height !== Math.round(rect.height * ratio);
}

function plotBounds(canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    left: 42,
    top: 14,
    right: rect.width - 10,
    bottom: rect.height - 24,
    width: rect.width,
    height: rect.height,
  };
}

function drawPlotFrame(ctx, canvas, bounds) {
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, bounds.width, bounds.height);
  ctx.strokeStyle = "#d8ded6";
  ctx.lineWidth = 1;
  ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
}

function drawTimeseriesPlot(ctx, canvas, bounds, plot) {
  const visibleSeries = (plot.series || []).filter((series) => isPlotSourceVisible(plot.id, series.key));
  const xRange = plot.time_range_s || [0, 1];
  const yRange = resolveTimeseriesYRange(plot, visibleSeries);
  drawTimeOverlays(ctx, bounds, xRange, plot.overlays || []);
  drawGrid(ctx, bounds);

  visibleSeries.forEach((series) => {
    drawLine(
      ctx,
      series.time_s || [],
      series.values || [],
      bounds,
      xRange,
      yRange,
      series.color,
    );
  });
}

function drawTimeseriesTracker(ctx, bounds, plot) {
  const visibleSeries = (plot.series || []).filter((series) => isPlotSourceVisible(plot.id, series.key));
  const xRange = plot.time_range_s || [0, 1];
  const yRange = resolveTimeseriesYRange(plot, visibleSeries);
  const tracker = clampTracker(plot.id, xRange[0], xRange[1]);
  drawTimeTracker(ctx, bounds, tracker, xRange);
  drawTimeseriesTrackerPoints(ctx, bounds, visibleSeries, tracker, xRange, yRange);
  updatePlotTime(plot.id, tracker);
}

function drawLocalPositionPlot(ctx, canvas, bounds, plot) {
  const visibleTraces = (plot.traces || []).filter((trace) => isPlotSourceVisible(plot.id, trace.key));
  const range = resolveLocalPositionRange(plot, visibleTraces, bounds);
  drawGrid(ctx, bounds);

  visibleTraces.forEach((trace) => {
    if (trace.marker_only) {
      drawScatter(ctx, trace.x || [], trace.y || [], bounds, range.x, range.y, trace.color);
    } else {
      drawLine(
        ctx,
        trace.x || [],
        trace.y || [],
        bounds,
        range.x,
        range.y,
        trace.color,
      );
    }
  });
}

function drawLocalPositionTracker(ctx, bounds, plot) {
  const visibleTraces = (plot.traces || []).filter((trace) => isPlotSourceVisible(plot.id, trace.key));
  const range = resolveLocalPositionRange(plot, visibleTraces, bounds);
  const tracker = clampTracker(plot.id, plot.time_range_s?.[0] || 0, plot.time_range_s?.[1] || 0);
  visibleTraces.forEach((trace) => {
    const marker = sampleTraceAtTime(trace, tracker);
    if (!marker) return;
    drawMarker(
      ctx,
      xToCanvas(marker.x, bounds, range.x),
      yToCanvas(marker.y, bounds, range.y),
      trace.color,
      trace.marker_only,
    );
  });
  updatePlotTime(plot.id, tracker);
}

function drawSpectrogramPlot(ctx, canvas, bounds, plot) {
  const xRange = plot.time_range_s || numericRange(plot.time_s || []);
  const yRange = plot.frequency_range_hz || numericRange(plot.frequencies_hz || []);
  drawSpectrogramImage(ctx, bounds, plot, xRange, yRange);
  drawGrid(ctx, bounds);
}

function drawSpectrogramTracker(ctx, bounds, plot) {
  const xRange = plot.time_range_s || numericRange(plot.time_s || []);
  const tracker = clampTracker(plot.id, xRange[0], xRange[1]);
  drawTimeTracker(ctx, bounds, tracker, xRange);
  updatePlotTime(plot.id, tracker);
}

function drawSpectrogramImage(ctx, bounds, plot, xRange, yRange) {
  const times = plot.time_s || [];
  const frequencies = plot.frequencies_hz || [];
  const values = plot.values_db || [];
  if (!times.length || !frequencies.length || !values.length) return;

  const [valueMin, valueMax] = plot.value_range_db || numericRange(values.flat());
  ctx.save();
  for (let timeIndex = 0; timeIndex < times.length; timeIndex += 1) {
    const time0 = spectrogramCellLower(times, timeIndex, xRange);
    const time1 = spectrogramCellUpper(times, timeIndex, xRange);
    const x0 = xToCanvas(time0, bounds, xRange);
    const x1 = xToCanvas(time1, bounds, xRange);
    for (let frequencyIndex = 0; frequencyIndex < frequencies.length; frequencyIndex += 1) {
      const row = values[frequencyIndex] || [];
      const value = row[timeIndex];
      if (!Number.isFinite(value)) continue;
      const freq0 = spectrogramCellLower(frequencies, frequencyIndex, yRange);
      const freq1 = spectrogramCellUpper(frequencies, frequencyIndex, yRange);
      const y0 = yToCanvas(freq0, bounds, yRange);
      const y1 = yToCanvas(freq1, bounds, yRange);
      ctx.fillStyle = viridisColor(value, valueMin, valueMax);
      ctx.fillRect(
        Math.floor(Math.min(x0, x1)),
        Math.floor(Math.min(y0, y1)),
        Math.ceil(Math.abs(x1 - x0)) + 1,
        Math.ceil(Math.abs(y1 - y0)) + 1,
      );
    }
  }
  ctx.restore();
}

function drawTimeOverlays(ctx, bounds, xRange, overlays) {
  overlays.forEach((overlay) => {
    const start = Number(overlay.start_s);
    const end = Number(overlay.end_s ?? overlay.start_s);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    const x1 = xToCanvas(start, bounds, xRange);
    const x2 = xToCanvas(end, bounds, xRange);
    ctx.save();
    ctx.globalAlpha = Number(overlay.alpha) || 0.12;
    ctx.fillStyle = overlay.color || "#b7c0b5";
    const top = overlay.band === "bottom"
      ? bounds.bottom - Math.min(40, bounds.bottom - bounds.top)
      : bounds.top;
    const bottom = bounds.bottom;
    if (Math.abs(x2 - x1) < 1) {
      ctx.fillRect(x1, top, 1, bottom - top);
    } else {
      ctx.fillRect(Math.min(x1, x2), top, Math.max(1, Math.abs(x2 - x1)), bottom - top);
    }
    ctx.restore();
  });
}

function drawGrid(ctx, bounds) {
  ctx.save();
  ctx.strokeStyle = "#eef1ec";
  ctx.lineWidth = 1;
  for (let index = 1; index < 4; index += 1) {
    const y = bounds.top + ((bounds.bottom - bounds.top) * index) / 4;
    ctx.beginPath();
    ctx.moveTo(bounds.left, y);
    ctx.lineTo(bounds.right, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawLine(ctx, xs, ys, bounds, xRange, yRange, color) {
  if (!xs.length || !ys.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  xs.forEach((xValue, index) => {
    const yValue = ys[index];
    if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
    const x = xToCanvas(xValue, bounds, xRange);
    const y = yToCanvas(yValue, bounds, yRange);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  if (started) ctx.stroke();
  ctx.restore();
}

function drawScatter(ctx, xs, ys, bounds, xRange, yRange, color) {
  if (!xs.length || !ys.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  xs.forEach((xValue, index) => {
    const yValue = ys[index];
    if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
    const x = xToCanvas(xValue, bounds, xRange);
    const y = yToCanvas(yValue, bounds, yRange);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function drawTimeTracker(ctx, bounds, time, xRange) {
  const x = xToCanvas(time, bounds, xRange);
  ctx.save();
  ctx.strokeStyle = "#202620";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, bounds.top);
  ctx.lineTo(x, bounds.bottom);
  ctx.stroke();
  ctx.restore();
}

function drawTimeseriesTrackerPoints(ctx, bounds, seriesList, tracker, xRange, yRange) {
  seriesList.forEach((series) => {
    const value = sampleSeriesAtTime(series, tracker);
    if (value == null) return;
    drawMarker(ctx, xToCanvas(tracker, bounds, xRange), yToCanvas(value, bounds, yRange), series.color);
  });
}

function drawMarker(ctx, x, y, color, filled = false) {
  ctx.save();
  ctx.fillStyle = filled ? color : "#ffffff";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function canvasTimeFromEvent(canvas, plot, event) {
  const bounds = plotBounds(canvas);
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  if (x < bounds.left || x > bounds.right) return null;
  const [start, end] = plot.time_range_s || [0, 0];
  const ratio = (x - bounds.left) / Math.max(1, bounds.right - bounds.left);
  return start + ratio * (end - start);
}

function updatePlotReadout(plot) {
  const readout = document.getElementById(`plotReadout-${plot.id}`);
  if (!readout) return;
  const tracker = state.plotTrackers[plot.id] ?? 0;

  if (plot.kind === "local_position") {
    readout.innerHTML = localPositionReadout(plot, tracker);
  } else if (plot.kind === "spectrogram") {
    readout.innerHTML = spectrogramReadout(plot, tracker);
  } else {
    readout.innerHTML = timeseriesReadout(plot, tracker);
  }
}

function timeseriesReadout(plot, tracker) {
  const rows = (plot.series || [])
    .filter((series) => isPlotSourceVisible(plot.id, series.key))
    .map((series) => {
      const value = sampleSeriesAtTime(series, tracker);
      const suffix = series.unit ? ` ${series.unit}` : "";
      return readoutRow(series.label, value == null ? "n/a" : `${formatNumber(value)}${suffix}`);
    });
  return rows.length ? rows.join("") : readoutRow("Visible series", "none");
}

function localPositionReadout(plot, tracker) {
  const samples = {};
  const rows = (plot.traces || [])
    .filter((trace) => isPlotSourceVisible(plot.id, trace.key))
    .map((trace) => {
      const sample = sampleTraceAtTime(trace, tracker);
      samples[trace.key] = sample;
      if (!sample) return readoutRow(trace.label, "n/a");
      const zText = sample.z == null ? "" : `, z=${formatNumber(sample.z)}`;
      return readoutRow(trace.label, `plot x=${formatNumber(sample.x)}, plot y=${formatNumber(sample.y)}${zText}`);
    });

  const current = samples.position;
  const setpoint = samples.setpoint;
  if (current && setpoint) {
    const dz = current.z == null || setpoint.z == null ? "" : `, dz=${formatNumber(setpoint.z - current.z)}`;
    rows.push(readoutRow("Setpoint delta", `dx=${formatNumber(setpoint.x - current.x)}, dy=${formatNumber(setpoint.y - current.y)}${dz}`));
  }

  const missionSetpoint = samples.position_setpoints;
  if (current && missionSetpoint) {
    const horizontal = Math.hypot(missionSetpoint.x - current.x, missionSetpoint.y - current.y);
    const vertical = current.z == null || missionSetpoint.z == null
      ? "n/a"
      : formatNumber(missionSetpoint.z - current.z);
    rows.push(readoutRow("Mission distance", `horizontal=${formatNumber(horizontal)} m, vertical=${vertical}${vertical === "n/a" ? "" : " m"}`));
  }

  return rows.length ? rows.join("") : readoutRow("Visible traces", "none");
}

function spectrogramReadout(plot, tracker) {
  const timeIndex = nearestIndex(plot.time_s || [], tracker);
  if (timeIndex < 0) return readoutRow("Top peaks", "n/a");

  const peaks = spectrogramPeaksAt(plot, timeIndex);
  const peakText = peaks.length
    ? peaks.map((peak) => `${formatNumber(peak.frequency)} Hz (${formatNumber(peak.value)} dB)`).join(", ")
    : "n/a";
  const rows = [
    readoutRow("Top peaks", peakText),
    readoutRow("Spectrogram time", formatLogTime((plot.time_s || [])[timeIndex])),
  ];
  if (Number.isFinite(plot.sampling_frequency_hz)) {
    rows.push(readoutRow("Sample rate", `${formatNumber(plot.sampling_frequency_hz)} Hz`));
  }
  return rows.join("");
}

function readoutRow(label, value) {
  return `<div class="plot-readout-row"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>`;
}

function sampleTraceAtTime(trace, time) {
  const mode = trace.marker_only ? "previous" : "linear";
  const x = sampleSeriesAtTime({ time_s: trace.time_s, values: trace.x }, time, mode);
  const y = sampleSeriesAtTime({ time_s: trace.time_s, values: trace.y }, time, mode);
  if (x == null || y == null) return null;
  const z = trace.z ? sampleSeriesAtTime(trace.z, time, mode) : null;
  return { x, y, z };
}

function sampleSeriesAtTime(series, time, mode = "linear") {
  const times = series.time_s || [];
  const values = series.values || [];
  if (!times.length || !values.length) return null;
  if (time <= times[0]) return values[0];
  if (time >= times[times.length - 1]) return values[values.length - 1];

  let low = 0;
  let high = times.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (times[mid] === time) return values[mid];
    if (times[mid] < time) low = mid + 1;
    else high = mid - 1;
  }

  const before = Math.max(0, low - 1);
  const after = Math.min(times.length - 1, low);
  const t0 = times[before];
  const t1 = times[after];
  const v0 = values[before];
  const v1 = values[after];
  if (mode === "previous") {
    return v0;
  }
  if (mode === "nearest") {
    return Math.abs(time - t0) <= Math.abs(t1 - time) ? v0 : v1;
  }
  if (t1 === t0) return v0;
  const ratio = (time - t0) / (t1 - t0);
  return v0 + (v1 - v0) * ratio;
}

function numericRange(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (!numeric.length) return [0, 1];
  let min = Math.min(...numeric);
  let max = Math.max(...numeric);
  if (min === max) {
    const pad = Math.max(1, Math.abs(min) * 0.1);
    min -= pad;
    max += pad;
  }
  const padding = (max - min) * 0.08;
  return [min - padding, max + padding];
}

function resolveTimeseriesYRange(plot, visibleSeries) {
  if (Array.isArray(plot.y_range) && plot.y_range.length >= 2) {
    return plot.y_range;
  }

  const cache = state.plotRenderMetrics[plot.id] || {};
  const visibilityKey = visibleSeries.map((series) => series.key).join("|");
  if (cache.yRangeKey !== visibilityKey) {
    cache.yRangeKey = visibilityKey;
    cache.yRange = numericRange(visibleSeries.flatMap((series) => series.values || []));
    state.plotRenderMetrics[plot.id] = cache;
  }
  return cache.yRange || [0, 1];
}

function resolveLocalPositionRange(plot, visibleTraces, bounds) {
  const cache = state.plotRenderMetrics[plot.id] || {};
  const visibilityKey = visibleTraces.map((trace) => trace.key).join("|");
  const dimensionsKey = `${bounds.right - bounds.left}:${bounds.bottom - bounds.top}`;
  const rangeKey = `${visibilityKey}:${dimensionsKey}`;
  if (cache.localPositionRangeKey !== rangeKey) {
    cache.localPositionRangeKey = rangeKey;
    cache.localPositionRange = localPositionRange(plot, visibleTraces, bounds);
    state.plotRenderMetrics[plot.id] = cache;
  }
  return cache.localPositionRange;
}

function xyRange(traces, bounds) {
  const xRange = numericRange(traces.flatMap((trace) => trace.x || []));
  const yRange = numericRange(traces.flatMap((trace) => trace.y || []));
  const plotWidth = Math.max(1, bounds.right - bounds.left);
  const plotHeight = Math.max(1, bounds.bottom - bounds.top);
  const xSpan = xRange[1] - xRange[0];
  const ySpan = yRange[1] - yRange[0];
  const targetYSpan = xSpan * (plotHeight / plotWidth);
  const targetXSpan = ySpan * (plotWidth / plotHeight);

  if (targetYSpan > ySpan) {
    const center = (yRange[0] + yRange[1]) / 2;
    yRange[0] = center - targetYSpan / 2;
    yRange[1] = center + targetYSpan / 2;
  } else if (targetXSpan > xSpan) {
    const center = (xRange[0] + xRange[1]) / 2;
    xRange[0] = center - targetXSpan / 2;
    xRange[1] = center + targetXSpan / 2;
  }

  return { x: xRange, y: yRange };
}

function localPositionRange(plot, visibleTraces, bounds) {
  const scaleTrace = (plot.traces || []).find((trace) => trace.key === plot.scale_from);
  const sourceTraces = scaleTrace ? [scaleTrace] : visibleTraces;
  const xValues = sourceTraces.flatMap((trace) => trace.x || []);
  const yValues = sourceTraces.flatMap((trace) => trace.y || []);
  if (!xValues.length || !yValues.length) {
    return xyRange(visibleTraces, bounds);
  }

  let minX = Math.min(...xValues);
  let maxX = Math.max(...xValues);
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  let xDiff = Math.max(maxX - minX, Number(plot.min_range) || 5);
  let yDiff = Math.max(maxY - minY, Number(plot.min_range) || 5);
  const xCenter = (minX + maxX) / 2;
  const yCenter = (minY + maxY) / 2;
  const aspect = Math.max(1, bounds.right - bounds.left) / Math.max(1, bounds.bottom - bounds.top);

  if (aspect > xDiff / yDiff) {
    xDiff = yDiff * aspect;
  } else {
    yDiff = xDiff / aspect;
  }

  const zoom = Number(plot.zoom_out_factor) || 1.3;
  return {
    x: [xCenter - (xDiff / 2) * zoom, xCenter + (xDiff / 2) * zoom],
    y: [yCenter - (yDiff / 2) * zoom, yCenter + (yDiff / 2) * zoom],
  };
}

function spectrogramCellLower(values, index, range) {
  if (index <= 0) {
    const width = values.length > 1 ? values[1] - values[0] : range[1] - range[0];
    return Math.max(range[0], values[0] - width / 2);
  }
  return (values[index - 1] + values[index]) / 2;
}

function spectrogramCellUpper(values, index, range) {
  if (index >= values.length - 1) {
    const width = values.length > 1 ? values[index] - values[index - 1] : range[1] - range[0];
    return Math.min(range[1], values[index] + width / 2);
  }
  return (values[index] + values[index + 1]) / 2;
}

function spectrogramPeaksAt(plot, timeIndex) {
  const frequencies = plot.frequencies_hz || [];
  const values = plot.values_db || [];
  const candidates = [];
  for (let index = 1; index < frequencies.length - 1; index += 1) {
    const value = values[index]?.[timeIndex];
    const previous = values[index - 1]?.[timeIndex];
    const next = values[index + 1]?.[timeIndex];
    if (!Number.isFinite(value) || !Number.isFinite(previous) || !Number.isFinite(next)) continue;
    if (frequencies[index] <= 0) continue;
    if (value >= previous && value >= next) {
      candidates.push({ frequency: frequencies[index], value });
    }
  }

  const fallback = candidates.length ? candidates : frequencies
    .map((frequency, index) => ({ frequency, value: values[index]?.[timeIndex] }))
    .filter((peak) => peak.frequency > 0 && Number.isFinite(peak.value));

  return fallback
    .sort((left, right) => right.value - left.value)
    .slice(0, 3);
}

function nearestIndex(values, target) {
  if (!values.length || !Number.isFinite(target)) return -1;
  let bestIndex = 0;
  let bestDistance = Math.abs(values[0] - target);
  for (let index = 1; index < values.length; index += 1) {
    const distance = Math.abs(values[index] - target);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return bestIndex;
}

function viridisColor(value, min, max) {
  const stops = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ];
  const span = max - min || 1;
  const ratio = Math.min(Math.max((value - min) / span, 0), 1);
  for (let index = 1; index < stops.length; index += 1) {
    const [stopRatio, stopColor] = stops[index];
    const [previousRatio, previousColor] = stops[index - 1];
    if (ratio <= stopRatio) {
      const local = (ratio - previousRatio) / Math.max(stopRatio - previousRatio, 0.0001);
      const color = stopColor.map((channel, channelIndex) => (
        Math.round(previousColor[channelIndex] + (channel - previousColor[channelIndex]) * local)
      ));
      return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
  }
  return "rgb(253, 231, 37)";
}

function xToCanvas(value, bounds, range) {
  const span = range[1] - range[0] || 1;
  return bounds.left + ((value - range[0]) / span) * (bounds.right - bounds.left);
}

function yToCanvas(value, bounds, range) {
  const span = range[1] - range[0] || 1;
  return bounds.bottom - ((value - range[0]) / span) * (bounds.bottom - bounds.top);
}

function isPlotSourceVisible(plotId, sourceKey) {
  return !state.hiddenPlotSeries[`${plotId}:${sourceKey}`];
}

function clampTracker(plotId, start, end) {
  const current = Number(state.plotTrackers[plotId]);
  const shared = Number(state.sharedPlotTracker);
  const fallback = Number.isFinite(shared) ? shared : Number.isFinite(start) ? start : 0;
  const value = Number.isFinite(current) ? current : fallback;
  const clamped = clampTime(value, start, end);
  state.plotTrackers[plotId] = clamped;
  return clamped;
}

function clampTime(value, start, end) {
  const lower = Number.isFinite(start) ? start : 0;
  const upper = Number.isFinite(end) ? end : lower;
  return Math.min(Math.max(value, lower), upper);
}

function updatePlotTime(plotId, tracker) {
  const time = document.getElementById(`plotTime-${plotId}`);
  if (time) time.textContent = formatLogTime(tracker);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(3);
  }
  return String(Math.round(value * 10000) / 10000);
}

function renderFacts() {
  const payload = state.payload;
  const inventory = payload?.inventory || {};
  const assumptions = payload?.assumptions || {};
  const inputName = basename(payload?.inputs?.log_path || "");

  els.sidebarSummary.textContent = inputName || "Log loaded";
  const facts = [
    ["Firmware", inventory.firmware_version || "unknown"],
    ["Branch", inventory.firmware_branch || "unknown"],
    ["Git hash", inventory.git_hash || "unknown"],
    ["Airframe", formatAirframe(inventory.airframe)],
    ["Duration", formatDuration(inventory.duration_s)],
    ["Vehicle", assumptions.vehicle_type || "unknown"],
    ["Topics", String((inventory.available_topics || []).length)],
    ["Params", String(payload?.parameters?.summary?.total || 0)],
  ];

  els.factGrid.innerHTML = facts
    .map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`)
    .join("");
}

function formatAirframe(airframe) {
  if (!airframe) {
    return "unknown";
  }

  const label = [airframe.name, airframe.type].filter(Boolean).join(", ");
  if (label && airframe.id !== undefined && airframe.id !== null) {
    return `${label} (${airframe.id})`;
  }
  return label || String(airframe.id || "unknown");
}

function renderWarnings() {
  const messages = inventoryMessages(state.payload?.inventory || {});
  const query = els.messageSearch.value.trim().toLowerCase();
  const filter = els.messageFilter.value;
  const filtered = messages.filter((message) => {
    if (filter === "important" && !isImportantLevel(message.level)) return false;
    if (filter === "errors" && !isErrorLevel(message.level)) return false;
    if (!query) return true;
    return [
      message.time,
      message.level,
      message.message,
      message.source,
    ]
      .map((value) => String(value || "").toLowerCase())
      .some((value) => value.includes(query));
  });

  els.warningCount.textContent = `${filtered.length}/${messages.length}`;
  els.warningsList.innerHTML = filtered.length
    ? filtered.map(renderMessageRow).join("")
    : `<tr><td colspan="3" class="empty-state">No messages match the current filter.</td></tr>`;
}

function inventoryMessages(inventory) {
  const loggedMessages = inventory.logged_messages || [];
  if (loggedMessages.length) {
    return loggedMessages.map((message) => ({
      time: formatLogTime(message.time_s),
      level: String(message.level || "UNKNOWN").toUpperCase(),
      message: String(message.message || ""),
      source: message.source || "",
    }));
  }

  return (inventory.warnings || []).map((warning) => ({
    time: "",
    level: "WARNING",
    message: String(warning || ""),
    source: "warning",
  }));
}

function renderMessageRow(message) {
  const levelClass = messageLevelClass(message.level);
  return `
    <tr class="${levelClass}">
      <td>${escapeHtml(message.time)}</td>
      <td><span class="level-pill">${escapeHtml(message.level)}</span></td>
      <td>${escapeHtml(message.message)}</td>
    </tr>
  `;
}

function isImportantLevel(level) {
  return ["EMERGENCY", "ALERT", "CRITICAL", "ERROR", "WARNING"].includes(String(level || "").toUpperCase());
}

function isErrorLevel(level) {
  return ["EMERGENCY", "ALERT", "CRITICAL", "ERROR"].includes(String(level || "").toUpperCase());
}

function messageLevelClass(level) {
  const normalized = String(level || "").toLowerCase();
  if (["emergency", "alert", "critical", "error"].includes(normalized)) return "message-error";
  if (normalized === "warning") return "message-warning";
  return "message-info";
}

function renderParameters() {
  const rows = state.payload?.parameters?.rows || [];
  const query = els.parameterSearch.value.trim().toLowerCase();
  const status = els.parameterStatus.value;
  const hideRcCal = els.hideRcCal.checked;

  const filtered = rows.filter((row) => {
    if (hideRcCal && row.is_rc_or_cal) return false;
    if (status !== "all" && row.default_status !== status) return false;
    if (!query) return true;
    return [
      row.name,
      row.value,
      row.default,
      row.default_source,
      row.default_status,
      row.description,
      row.group,
      row.type,
    ]
      .map((value) => String(value || "").toLowerCase())
      .some((value) => value.includes(query));
  });

  els.parameterCount.textContent = `${filtered.length}/${rows.length}`;
  els.parameterRows.innerHTML = filtered.length
    ? filtered.map(renderParameterRow).join("")
    : `<tr><td colspan="4" class="empty-state">No parameters match.</td></tr>`;
}

function renderParameterRow(row) {
  const title = [
    row.description,
    row.long_description,
    row.min || row.max ? `Range: ${row.min || ""} to ${row.max || ""}` : "",
    row.default_source ? `Default source: ${row.default_source}` : "Default source: unknown",
  ]
    .filter(Boolean)
    .join("\n");
  const statusClass = row.default_status === "non_default"
    ? "non-default"
    : row.default_status;
  return `
    <tr title="${escapeAttr(title)}">
      <td><code>${escapeHtml(row.name)}</code></td>
      <td>${escapeHtml(formatParameterCell(row.value))}</td>
      <td>${escapeHtml(formatParameterCell(row.default))}</td>
      <td><span class="status-badge ${escapeAttr(statusClass)}">${escapeHtml(statusLabel(row.default_status))}</span></td>
    </tr>
  `;
}

function renderTopics() {
  const rows = state.payload?.topics || [];
  const query = els.topicSearch.value.trim().toLowerCase();
  const filtered = rows.filter((row) => {
    if (!query) return true;
    return [row.name, ...(row.fields || [])]
      .map((value) => String(value || "").toLowerCase())
      .some((value) => value.includes(query));
  });

  els.topicCount.textContent = `${filtered.length}/${rows.length}`;
  els.topicRows.innerHTML = filtered.length
    ? filtered.map(renderTopicRow).join("")
    : `<p class="empty-state">No topics match.</p>`;
}

function renderTopicRow(row) {
  const flags = [];
  if (row.missing_expected) flags.push("missing expected");
  if (row.has_derived_attitude_fields) flags.push("derived RPY");
  const meta = `${row.field_count} fields${flags.length ? ` · ${flags.join(" · ")}` : ""}`;
  const fields = (row.fields || [])
    .map((field) => `<span class="field-chip">${escapeHtml(field)}</span>`)
    .join("");
  return `
    <details class="topic-row">
      <summary>
        <span class="topic-name">${escapeHtml(row.name)}</span>
        <span class="topic-meta">${escapeHtml(meta)}</span>
      </summary>
      <div class="field-list">${fields || `<span class="empty-state">No fields inventoried.</span>`}</div>
    </details>
  `;
}

function renderMainSummary() {
  const payload = state.payload;
  const params = payload?.parameters?.summary || {};
  const inventory = payload?.inventory || {};
  const uploadText = payload?.upload?.log_path
    ? `<p>Uploaded log saved at <code>${escapeHtml(payload.upload.log_path)}</code>.</p>`
    : "";
  els.mainSummary.innerHTML = `
    <div class="summary-kpis">
      <div class="kpi"><strong>${escapeHtml(formatCell(inventory.duration_s || ""))}</strong><span>seconds</span></div>
      <div class="kpi"><strong>${escapeHtml(formatCell(params.total || 0))}</strong><span>parameters</span></div>
      <div class="kpi"><strong>${escapeHtml(formatCell(params.non_default || 0))}</strong><span>non-default</span></div>
      <div class="kpi"><strong>${escapeHtml(formatCell((inventory.available_topics || []).length))}</strong><span>topics</span></div>
    </div>
    ${uploadText}
    <p>Default comparison uses embedded ULog airframe defaults, then embedded system defaults, then the optional parameters XML fallback.</p>
  `;
}

function renderLogTags() {
  const logId = currentBrowseLogId();
  els.logTagPanel.hidden = !state.payload || !logId;
  if (els.logTagPanel.hidden) return;

  const assigned = state.currentLogTags || [];
  const available = (state.availableTags || [])
    .map((tag) => tag.name)
    .filter((tag) => !assigned.includes(tag));
  els.logTagStatus.textContent = state.tagStatus || "";
  els.logTagList.innerHTML = assigned.length
    ? assigned.map((tag) => `
      <button class="tag-pill" data-remove-log-tag="${escapeAttr(tag)}" type="button" title="Remove tag">
        ${escapeHtml(tag)} <span aria-hidden="true">x</span>
      </button>
    `).join("")
    : `<span class="empty-inline">No tags assigned.</span>`;
  els.logTagSelect.innerHTML = `
    <option value="">Add existing tag</option>
    ${available.map((tag) => `<option value="${escapeAttr(tag)}">${escapeHtml(tag)}</option>`).join("")}
  `;
  els.logTagAdd.disabled = !available.length;
}

function renderChangedParameters() {
  const rows = state.payload?.parameters?.changed || [];
  els.changedParameterRows.innerHTML = rows.length
    ? rows.map((row) => `
      <tr>
        <td>${escapeHtml(`${row.time_s}s`)}</td>
        <td><code>${escapeHtml(row.name)}</code></td>
        <td>${escapeHtml(formatCell(row.value))}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="3" class="empty-state">No in-flight parameter changes found.</td></tr>`;
}

function renderTimeline() {
  const rows = (state.payload?.timeline || []).slice(0, 80);
  els.timelineRows.innerHTML = rows.length
    ? rows.map((row) => `
      <div class="timeline-item">
        <span class="timeline-time">${escapeHtml(row.time_s == null ? "n/a" : `${row.time_s}s`)}</span>
        <span>${escapeHtml(timelineText(row))}</span>
      </div>
    `).join("")
    : `<p class="empty-state">No timeline events found.</p>`;
}

function timelineText(row) {
  if (row.topic && row.field) {
    return `${row.event}: ${row.topic}.${row.field} = ${formatCell(row.value)}`;
  }
  return row.details || row.event || "timeline event";
}

function setStatus(text) {
  els.runStatus.textContent = text;
}

function setAnalysisStatus(text) {
  els.analysisStatus.textContent = text;
}

function statusText(status, phase) {
  const labels = {
    queued: "Queued",
    running: phase || "Running",
    completed: "Complete",
    failed: "Failed",
  };
  return labels[status] || status || "Idle";
}

function formatEventTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function artifactUrl(path) {
  return `/artifacts?path=${encodeURIComponent(path)}`;
}

function stringOrNull(value) {
  const trimmed = String(value || "").trim();
  return trimmed || null;
}

function basename(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function formatDuration(value) {
  return value == null ? "unknown" : `${value}s`;
}

function formatLogTime(value) {
  if (value == null || value === "") return "";
  const totalSeconds = Number(value);
  if (!Number.isFinite(totalSeconds)) return "";
  const wholeSeconds = Math.floor(totalSeconds);
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const seconds = wholeSeconds % 60;
  return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function statusLabel(status) {
  return {
    default: "default",
    non_default: "non-default",
    unknown: "unknown",
  }[status] || "unknown";
}

function formatCell(value) {
  if (value == null || value === "") return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatParameterCell(value) {
  if (value == null || value === "") return "";
  if (typeof value === "object") return JSON.stringify(value);

  const text = String(value).trim();
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text)) {
    return String(value);
  }

  const numeric = Number(text);
  if (!Number.isFinite(numeric)) return String(value);

  const rounded = Math.round(numeric * 10000) / 10000;
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("\n", "&#10;");
}
