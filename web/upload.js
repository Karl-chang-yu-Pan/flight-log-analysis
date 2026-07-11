const uploadEls = {
  form: document.getElementById("uploadForm"),
  logFile: document.getElementById("logFile"),
  logPath: document.getElementById("logPath"),
  sourcePath: document.getElementById("sourcePath"),
  missionPath: document.getElementById("missionPath"),
  parametersXmlPath: document.getElementById("parametersXmlPath"),
  button: document.getElementById("uploadButton"),
  status: document.getElementById("uploadStatus"),
  progress: document.getElementById("uploadProgress"),
  progressBar: document.getElementById("uploadProgressBar"),
  error: document.getElementById("uploadError"),
};

uploadEls.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await uploadAndOpenReview();
});

async function uploadAndOpenReview() {
  const hasUpload = uploadEls.logFile.files && uploadEls.logFile.files.length > 0;
  const localLogPath = uploadEls.logPath.value.trim();
  if (!hasUpload && !localLogPath) {
    showUploadError("Choose a ULog file or provide a local ULog path.");
    return;
  }

  uploadEls.button.disabled = true;
  uploadEls.error.hidden = true;
  setUploadStatus(hasUpload ? "Uploading" : "Loading local log");

  try {
    const result = hasUpload
      ? await uploadLog(new FormData(uploadEls.form))
      : await indexLocalLog(localLogPath);
    const browseId = result?.browse?.log_id;
    if (!browseId) {
      throw new Error(result?.browse?.error || "The log was parsed but not indexed.");
    }
    setUploadProgress(100);
    setUploadStatus("Opening review");
    window.location.assign(`/review?browse_id=${encodeURIComponent(browseId)}`);
  } catch (error) {
    showUploadError(error.message);
    setUploadStatus("Failed");
    uploadEls.button.disabled = false;
    hideUploadProgressSoon();
  }
}

function uploadLog(formData) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    setUploadProgress(0);
    request.open("POST", "/api/upload-preparse");
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      setUploadProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
    });
    request.addEventListener("load", () => {
      let result;
      try {
        result = JSON.parse(request.responseText);
      } catch {
        reject(new Error("The server returned an invalid response."));
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(result.error || `HTTP ${request.status}`));
        return;
      }
      resolve(result);
    });
    request.addEventListener("error", () => reject(new Error("Upload failed.")));
    request.send(formData);
  });
}

async function indexLocalLog(logPath) {
  const response = await fetch("/api/preparse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      log_path: logPath,
      mission_path: valueOrNull(uploadEls.missionPath.value),
      source_path: valueOrNull(uploadEls.sourcePath.value),
      parameters_xml_path: valueOrNull(uploadEls.parametersXmlPath.value),
    }),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}

function setUploadStatus(message) {
  uploadEls.status.textContent = message;
}

function setUploadProgress(percent) {
  uploadEls.progress.hidden = false;
  uploadEls.progressBar.style.width = `${percent}%`;
}

function hideUploadProgressSoon() {
  window.setTimeout(() => {
    uploadEls.progress.hidden = true;
    uploadEls.progressBar.style.width = "0%";
  }, 500);
}

function showUploadError(message) {
  uploadEls.error.textContent = message;
  uploadEls.error.hidden = false;
}

function valueOrNull(value) {
  const clean = String(value || "").trim();
  return clean || null;
}
