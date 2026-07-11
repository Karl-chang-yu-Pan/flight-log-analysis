const browseState = {
  rows: [],
  tags: [],
  selectedTags: [],
  sort: "upload_date",
  direction: "desc",
  limit: 50,
  offset: 0,
  filtered: 0,
  total: 0,
  loadTimer: null,
};

const browseEls = {
  status: document.getElementById("browseStatus"),
  search: document.getElementById("browseSearch"),
  uploadStart: document.getElementById("uploadStart"),
  uploadEnd: document.getElementById("uploadEnd"),
  logStart: document.getElementById("logStart"),
  logEnd: document.getElementById("logEnd"),
  tagFilters: document.getElementById("tagFilters"),
  importFlightReview: document.getElementById("importFlightReview"),
  rows: document.getElementById("browseRows"),
  prevPage: document.getElementById("prevPage"),
  nextPage: document.getElementById("nextPage"),
  pageSummary: document.getElementById("pageSummary"),
};

loadStateFromUrl();
bindBrowseEvents();
refreshBrowse();

function bindBrowseEvents() {
  for (const input of [
    browseEls.search,
    browseEls.uploadStart,
    browseEls.uploadEnd,
    browseEls.logStart,
    browseEls.logEnd,
  ]) {
    input.addEventListener("input", () => {
      browseState.offset = 0;
      scheduleRefresh();
    });
  }

  browseEls.importFlightReview.addEventListener("click", importFlightReview);
  browseEls.prevPage.addEventListener("click", () => {
    browseState.offset = Math.max(0, browseState.offset - browseState.limit);
    refreshBrowse();
  });
  browseEls.nextPage.addEventListener("click", () => {
    if (browseState.offset + browseState.limit < browseState.filtered) {
      browseState.offset += browseState.limit;
      refreshBrowse();
    }
  });

  document.querySelectorAll(".sort-button").forEach((button) => {
    button.addEventListener("click", () => {
      const sort = button.dataset.sort || "upload_date";
      if (browseState.sort === sort) {
        browseState.direction = browseState.direction === "asc" ? "desc" : "asc";
      } else {
        browseState.sort = sort;
        browseState.direction = sort === "error_count" ? "desc" : "asc";
      }
      browseState.offset = 0;
      refreshBrowse();
    });
  });

  browseEls.tagFilters.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tag-filter]");
    if (!button) return;
    const tag = button.dataset.tagFilter;
    if (browseState.selectedTags.includes(tag)) {
      browseState.selectedTags = browseState.selectedTags.filter((item) => item !== tag);
    } else {
      browseState.selectedTags = [...browseState.selectedTags, tag];
    }
    browseState.offset = 0;
    refreshBrowse();
  });

}

function scheduleRefresh() {
  window.clearTimeout(browseState.loadTimer);
  browseState.loadTimer = window.setTimeout(refreshBrowse, 250);
}

async function refreshBrowse() {
  writeStateToUrl();
  setStatus("Loading");
  try {
    const [logs, tagsPayload] = await Promise.all([
      fetchJson(`/api/browse-logs?${browseQueryParams()}`),
      fetchJson("/api/browse-tags"),
    ]);
    browseState.rows = logs.rows || [];
    browseState.total = logs.total || 0;
    browseState.filtered = logs.filtered || 0;
    browseState.limit = logs.limit || browseState.limit;
    browseState.offset = logs.offset || 0;
    browseState.tags = tagsPayload.tags || [];
    renderTags();
    renderRows();
    renderPagination();
    setStatus(`${browseState.filtered} of ${browseState.total} logs`);
  } catch (error) {
    setStatus("Load failed");
    browseEls.rows.innerHTML = `<tr><td colspan="9" class="empty-state">${escapeHtml(error.message)}</td></tr>`;
  }
}

function browseQueryParams() {
  const params = new URLSearchParams();
  setParam(params, "search", browseEls.search.value.trim());
  setParam(params, "upload_start", browseEls.uploadStart.value);
  setParam(params, "upload_end", browseEls.uploadEnd.value);
  setParam(params, "log_start", browseEls.logStart.value);
  setParam(params, "log_end", browseEls.logEnd.value);
  browseState.selectedTags.forEach((tag) => params.append("tag", tag));
  setParam(params, "sort", browseState.sort);
  setParam(params, "direction", browseState.direction);
  setParam(params, "limit", String(browseState.limit));
  setParam(params, "offset", String(browseState.offset));
  return params.toString();
}

function renderTags() {
  browseEls.tagFilters.innerHTML = browseState.tags.length
    ? browseState.tags.map((tag) => {
      const selected = browseState.selectedTags.includes(tag.name);
      return `
        <button class="tag-filter ${selected ? "selected" : ""}" data-tag-filter="${escapeHtml(tag.name)}" type="button">
          ${escapeHtml(tag.name)}
          <span>${tag.log_count}</span>
        </button>
      `;
    }).join("")
    : `<span class="empty-inline">No tags</span>`;
}

function renderRows() {
  if (!browseState.rows.length) {
    browseEls.rows.innerHTML = `<tr><td colspan="9" class="empty-state">No logs match the current filters.</td></tr>`;
    return;
  }

  browseEls.rows.innerHTML = browseState.rows.map((row) => `
    <tr>
      <td><a class="table-link" href="/?browse_id=${encodeURIComponent(row.id)}">${escapeHtml(formatDate(row.upload_date) || "Open")}</a></td>
      <td>${escapeHtml(formatDate(row.log_date))}</td>
      <td>${renderAirframe(row)}</td>
      <td>${escapeHtml(row.hardware || "")}</td>
      <td>${escapeHtml(row.software_version || row.software || "")}</td>
      <td>${escapeHtml(formatDuration(row.duration_s))}</td>
      <td><span class="${row.error_count > 0 ? "error-count has-errors" : "error-count"}">${escapeHtml(String(row.error_count || 0))}</span></td>
      <td>${escapeHtml(formatFlightModes(row.flight_modes || []))}</td>
      <td>${renderRowTags(row)}</td>
    </tr>
  `).join("");
}

function renderAirframe(row) {
  const key = String(row.airframe_image_key || "AirframeUnknown");
  const filename = key.toLowerCase().endsWith(".svg") ? key : `${key}.svg`;
  const title = [
    row.vehicle_type,
    row.airframe_name,
    row.airframe_group,
    row.airframe_id ? `SYS_AUTOSTART ${row.airframe_id}` : "",
  ].filter(Boolean).join(" / ") || "Unknown airframe";
  return `
    <img
      class="airframe-thumb"
      src="/airframe_img/${encodeURIComponent(filename)}"
      alt=""
      title="${escapeHtml(title)}"
      loading="lazy"
      decoding="async">
  `;
}

function renderRowTags(row) {
  const assigned = row.tags || [];
  const tagPills = assigned.map((tag) => `
    <span class="tag-pill readonly">${escapeHtml(tag)}</span>
  `).join("");

  return `<div class="row-tags">${tagPills || `<span class="empty-inline">None</span>`}</div>`;
}

function renderPagination() {
  browseEls.prevPage.disabled = browseState.offset <= 0;
  browseEls.nextPage.disabled = browseState.offset + browseState.limit >= browseState.filtered;
  const start = browseState.filtered ? browseState.offset + 1 : 0;
  const end = Math.min(browseState.offset + browseState.limit, browseState.filtered);
  browseEls.pageSummary.textContent = `${start}-${end} of ${browseState.filtered}`;
}

async function importFlightReview() {
  browseEls.importFlightReview.disabled = true;
  setStatus("Importing Flight Review logs");
  try {
    const result = await fetchJson("/api/browse-import-flight-review", { method: "POST" });
    setStatus(`Imported ${result.imported || 0} logs`);
    await refreshBrowse();
  } catch (error) {
    setStatus(error.message);
  } finally {
    browseEls.importFlightReview.disabled = false;
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}

function loadStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  browseEls.search.value = params.get("search") || "";
  browseEls.uploadStart.value = params.get("upload_start") || "";
  browseEls.uploadEnd.value = params.get("upload_end") || "";
  browseEls.logStart.value = params.get("log_start") || "";
  browseEls.logEnd.value = params.get("log_end") || "";
  browseState.selectedTags = (params.get("tags") || "").split(",").map((tag) => tag.trim()).filter(Boolean);
  browseState.sort = params.get("sort") || browseState.sort;
  browseState.direction = params.get("direction") || browseState.direction;
}

function writeStateToUrl() {
  const params = new URLSearchParams();
  setParam(params, "search", browseEls.search.value.trim());
  setParam(params, "upload_start", browseEls.uploadStart.value);
  setParam(params, "upload_end", browseEls.uploadEnd.value);
  setParam(params, "log_start", browseEls.logStart.value);
  setParam(params, "log_end", browseEls.logEnd.value);
  setParam(params, "tags", browseState.selectedTags.join(","));
  setParam(params, "sort", browseState.sort === "upload_date" ? "" : browseState.sort);
  setParam(params, "direction", browseState.direction === "desc" ? "" : browseState.direction);
  const nextUrl = `${window.location.pathname}${params.toString() ? `?${params}` : ""}`;
  window.history.replaceState(null, "", nextUrl);
}

function setParam(params, name, value) {
  if (value) params.set(name, value);
}

function setStatus(message) {
  browseEls.status.textContent = message;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 16);
  return date.toLocaleString([], {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(value) {
  if (value == null || value === "") return "";
  const seconds = Math.round(Number(value));
  if (!Number.isFinite(seconds)) return "";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${rest}s`;
  return `${rest}s`;
}

function formatFlightModes(modes) {
  return modes.map((mode) => `nav ${mode}`).join(", ");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
