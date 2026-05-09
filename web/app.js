const state = {
  payload: null,
  sidebarCollapsed: false,
};

const els = {
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
  parameterSearch: document.getElementById("parameterSearch"),
  parameterStatus: document.getElementById("parameterStatus"),
  hideRcCal: document.getElementById("hideRcCal"),
  parameterRows: document.getElementById("parameterRows"),
  parameterCount: document.getElementById("parameterCount"),
  topicSearch: document.getElementById("topicSearch"),
  topicRows: document.getElementById("topicRows"),
  topicCount: document.getElementById("topicCount"),
  mainSummary: document.getElementById("mainSummary"),
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
els.topicSearch.addEventListener("input", renderTopics);

els.sidebarToggle.addEventListener("click", () => {
  state.sidebarCollapsed = !state.sidebarCollapsed;
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
    renderAll();
    setStatus("Loaded");
  } catch (error) {
    setStatus("Failed");
    els.mainSummary.innerHTML = `<p class="message-item">${escapeHtml(error.message)}</p>`;
  } finally {
    hideUploadProgressSoon();
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
    ["Git hash", inventory.git_hash || "unknown"],
    ["Duration", formatDuration(inventory.duration_s)],
    ["Vehicle", assumptions.vehicle_type || "unknown"],
    ["Topics", String((inventory.available_topics || []).length)],
    ["Params", String(payload?.parameters?.summary?.total || 0)],
  ];

  els.factGrid.innerHTML = facts
    .map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`)
    .join("");
}

function renderWarnings() {
  const warnings = state.payload?.inventory?.warnings || [];
  els.warningCount.textContent = String(warnings.length);
  els.warningsList.innerHTML = warnings.length
    ? warnings.map((warning) => `<div class="message-item">${escapeHtml(warning)}</div>`).join("")
    : `<p class="empty-state">No warnings reported by the parser.</p>`;
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
      <td>${escapeHtml(formatCell(row.value))}</td>
      <td>${escapeHtml(formatCell(row.default))}</td>
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
