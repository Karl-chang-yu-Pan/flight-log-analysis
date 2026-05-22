const state = {
  payload: null,
  analysisRunId: null,
  analysisPollTimer: null,
  sidebarCollapsed: false,
};

const els = {
  appShell: document.getElementById("appShell"),
  form: document.getElementById("preparseForm"),
  logFile: document.getElementById("logFile"),
  runStatus: document.getElementById("runStatus"),
  uploadProgress: document.getElementById("uploadProgress"),
  uploadProgressBar: document.getElementById("uploadProgressBar"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebarToggle"),
  sidebarSummary: document.getElementById("sidebarSummary"),
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
  analysisQuestion: document.getElementById("analysisQuestion"),
  analysisButton: document.getElementById("analysisButton"),
  analysisStatus: document.getElementById("analysisStatus"),
  analysisProgress: document.getElementById("analysisProgress"),
  analysisReport: document.getElementById("analysisReport"),
  changedParameterRows: document.getElementById("changedParameterRows"),
  timelineRows: document.getElementById("timelineRows"),
};

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await preparseLog();
});

els.parameterSearch.addEventListener("input", renderParameters);
els.parameterStatus.addEventListener("change", renderParameters);
els.hideRcCal.addEventListener("change", renderParameters);
els.messageSearch.addEventListener("input", renderWarnings);
els.messageFilter.addEventListener("change", renderWarnings);
els.topicSearch.addEventListener("input", renderTopics);
els.analysisButton.addEventListener("click", startAnalysis);

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

async function preparseLog() {
  setStatus("Pre-parsing...");
  const formData = new FormData(els.form);
  const hasUpload = els.logFile.files && els.logFile.files.length > 0;

  try {
    const result = hasUpload
      ? await uploadAndPreparse(formData)
      : await preparseLocalPath(formData);
    state.payload = result;
    resetAnalysis();
    renderAll();
    setStatus("Loaded");
  } catch (error) {
    setStatus("Failed");
    els.mainSummary.innerHTML = `<p class="message-item">${escapeHtml(error.message)}</p>`;
  } finally {
    hideUploadProgressSoon();
  }
}

async function startAnalysis() {
  if (!state.payload) {
    setAnalysisStatus("Load a log first");
    return;
  }

  const inputs = state.payload.inputs || {};
  const question = els.analysisQuestion.value.trim()
    || "Analyze this flight log and identify the most likely root causes.";
  const payload = {
    log_path: inputs.log_path,
    mission_path: inputs.mission_path,
    source_path: inputs.source_path,
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

async function preparseLocalPath(formData) {
  const payload = {
    log_path: stringOrNull(formData.get("log_path")),
    mission_path: stringOrNull(formData.get("mission_path")),
    source_path: stringOrNull(formData.get("source_path")),
    parameters_xml_path: stringOrNull(formData.get("parameters_xml_path")),
  };

  if (!payload.log_path) {
    throw new Error("Choose a ULog file or provide a local ULog path.");
  }

  const response = await fetch("/api/preparse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}

function uploadAndPreparse(formData) {
  return new Promise((resolve, reject) => {
    showUploadProgress(0);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload-preparse");
    xhr.responseType = "json";

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        showUploadProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
      }
    });

    xhr.addEventListener("load", () => {
      showUploadProgress(100);
      const result = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(result);
      } else {
        reject(new Error(result.error || `HTTP ${xhr.status}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Upload failed."));
    });

    xhr.send(formData);
  });
}

function renderAll() {
  renderFacts();
  renderWarnings();
  renderParameters();
  renderTopics();
  renderMainSummary();
  renderChangedParameters();
  renderTimeline();
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

function showUploadProgress(percent) {
  els.uploadProgress.hidden = false;
  els.uploadProgressBar.style.width = `${percent}%`;
}

function hideUploadProgressSoon() {
  window.setTimeout(() => {
    els.uploadProgress.hidden = true;
    els.uploadProgressBar.style.width = "0%";
  }, 800);
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
