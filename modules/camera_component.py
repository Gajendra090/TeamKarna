"""Edge-compatible persistent WebRTC camera component for KNOWLEDGE X Scan Documents."""

from __future__ import annotations

import base64
import binascii

import streamlit as st

_CAMERA_HTML = """
<div class="kx-cam-wrap">
  <div class="kx-cam-toolbar">
    <select class="kx-cam-select" id="kxCamSelect" aria-label="Select camera device">
      <option value="">Default Camera</option>
    </select>
    <span class="kx-cam-status" id="kxCamStatus">Camera ready</span>
  </div>

  <div class="kx-cam-stage" id="kxCamStage">
    <video class="kx-cam-video" id="kxCamVideo" autoplay playsinline muted></video>
    <div class="kx-cam-placeholder" id="kxCamPlaceholder">
      Click <b>Start Camera</b> to begin live webcam preview.
    </div>
  </div>

  <div class="kx-cam-error" id="kxCamError" hidden></div>

  <div class="kx-cam-actions">
    <button type="button" class="kx-cam-btn kx-cam-btn-primary" id="kxBtnStart">Start Camera</button>
    <button type="button" class="kx-cam-btn kx-cam-btn-primary" id="kxBtnCapture" hidden>Take Photo</button>
    <button type="button" class="kx-cam-btn" id="kxBtnRestart" hidden>Restart Camera</button>
    <button type="button" class="kx-cam-btn kx-cam-btn-danger" id="kxBtnStop" hidden>Stop Camera</button>
  </div>
</div>
"""

_CAMERA_CSS = """
.kx-cam-wrap {
  background: #111827;
  border: 1px dashed #334155;
  border-radius: 10px;
  padding: 0.75rem;
  color: #f1f5f9;
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  box-sizing: border-box;
}
.kx-cam-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 0.6rem;
  flex-wrap: wrap;
}
.kx-cam-select {
  flex: 1 1 180px;
  min-width: 0;
  background: #0f172a;
  color: #f1f5f9;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 0.35rem 0.55rem;
  font-size: 0.82rem;
}
.kx-cam-status {
  font-size: 0.76rem;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 999px;
  background: #172033;
  border: 1px solid #334155;
  color: #94a3b8;
  white-space: nowrap;
}
.kx-cam-status.live {
  background: rgba(52, 211, 153, 0.14);
  border-color: rgba(52, 211, 153, 0.45);
  color: #6ee7b7;
}
.kx-cam-status.error {
  background: rgba(248, 113, 113, 0.14);
  border-color: rgba(248, 113, 113, 0.45);
  color: #fca5a5;
}
.kx-cam-stage {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #0b1120;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.kx-cam-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: none;
}
.kx-cam-video.active {
  display: block;
}
.kx-cam-placeholder {
  padding: 1rem;
  text-align: center;
  font-size: 0.88rem;
  color: #94a3b8;
  line-height: 1.45;
}
.kx-cam-error {
  margin-top: 0.55rem;
  padding: 0.55rem 0.75rem;
  border-radius: 8px;
  background: rgba(248, 113, 113, 0.12);
  border: 1px solid rgba(248, 113, 113, 0.45);
  color: #fca5a5;
  font-size: 0.82rem;
  line-height: 1.4;
}
.kx-cam-actions {
  display: flex;
  gap: 8px;
  margin-top: 0.65rem;
  flex-wrap: wrap;
}
.kx-cam-btn {
  flex: 1 1 110px;
  padding: 0.48rem 0.8rem;
  border-radius: 8px;
  border: 1px solid #334155;
  background: #172033;
  color: #f1f5f9;
  font-size: 0.86rem;
  font-weight: 600;
  cursor: pointer;
}
.kx-cam-btn:hover {
  border-color: #38bdf8;
}
.kx-cam-btn-primary {
  background: linear-gradient(135deg, #0284c7, #4f46e5);
  border-color: transparent;
  color: #ffffff;
}
.kx-cam-btn-danger:hover {
  border-color: #f87171;
  color: #fecaca;
}
"""

_CAMERA_JS = """
export default function(component) {
  const { parentElement, setTriggerValue } = component;
  const video = parentElement.querySelector("#kxCamVideo");
  const placeholder = parentElement.querySelector("#kxCamPlaceholder");
  const selectEl = parentElement.querySelector("#kxCamSelect");
  const statusEl = parentElement.querySelector("#kxCamStatus");
  const errorEl = parentElement.querySelector("#kxCamError");
  const btnStart = parentElement.querySelector("#kxBtnStart");
  const btnCapture = parentElement.querySelector("#kxBtnCapture");
  const btnRestart = parentElement.querySelector("#kxBtnRestart");
  const btnStop = parentElement.querySelector("#kxBtnStop");

  const STATE = (window.__kxCamState = window.__kxCamState || {
    stream: null,
    deviceId: "",
    starting: false,
    beforeUnloadBound: false,
  });

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = "kx-cam-status" + (kind ? " " + kind : "");
  }

  function showError(msg) {
    if (!msg) {
      errorEl.hidden = true;
      errorEl.textContent = "";
      return;
    }
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  function isStreamLive(stream) {
    if (!stream) return false;
    const tracks = stream.getVideoTracks ? stream.getVideoTracks() : [];
    return tracks.length > 0 && tracks.some((t) => t.readyState === "live");
  }

  function syncUi() {
    const live = isStreamLive(STATE.stream);
    if (live) {
      if (video.srcObject !== STATE.stream) {
        video.srcObject = STATE.stream;
      }
      video.play().catch(() => {});
      video.classList.add("active");
      placeholder.style.display = "none";
      btnStart.hidden = true;
      btnCapture.hidden = false;
      btnRestart.hidden = false;
      btnStop.hidden = false;
      const track = STATE.stream.getVideoTracks()[0];
      const label = (track && track.label) ? track.label : "Webcam";
      setStatus("Live: " + label, "live");
    } else {
      video.classList.remove("active");
      video.srcObject = null;
      placeholder.style.display = "block";
      btnStart.hidden = false;
      btnCapture.hidden = true;
      btnRestart.hidden = true;
      btnStop.hidden = true;
      if (!errorEl.hidden) {
        setStatus("Camera error", "error");
      } else {
        setStatus("Camera stopped", "");
      }
    }
  }

  async function refreshDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      const current = STATE.deviceId || selectEl.value || "";
      selectEl.innerHTML = "";
      const defOpt = document.createElement("option");
      defOpt.value = "";
      defOpt.textContent = cams.length ? "Auto / Default Camera" : "No camera detected";
      selectEl.appendChild(defOpt);
      cams.forEach((cam, idx) => {
        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.textContent = cam.label || ("Camera " + (idx + 1));
        selectEl.appendChild(opt);
      });
      if (current && cams.some((c) => c.deviceId === current)) {
        selectEl.value = current;
      } else if (cams.length === 1 && cams[0].deviceId) {
        selectEl.value = cams[0].deviceId;
        STATE.deviceId = cams[0].deviceId;
      }
    } catch (_) {}
  }

  function stopStream(silent) {
    if (STATE.stream) {
      try {
        STATE.stream.getTracks().forEach((t) => {
          t.onended = null;
          t.stop();
        });
      } catch (_) {}
      STATE.stream = null;
    }
    if (!silent) {
      showError("");
    }
    syncUi();
  }

  function formatMediaError(err) {
    const name = (err && err.name) || "Error";
    const msg = (err && err.message) || "";
    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      return "Camera permission was denied. Click the camera/lock icon in Edge's address bar, allow Camera for localhost:8501, then click Start Camera.";
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return "Camera is busy or locked by another app/tab (NotReadableError). Close other apps or browser tabs using HP Wide Vision HD Camera, then click Start Camera.";
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return "No webcam device was found (NotFoundError). Check Windows Camera privacy settings and device connection.";
    }
    if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
      return "Selected camera rejected the requested resolution (OverconstrainedError).";
    }
    if (name === "AbortError") {
      return "Camera timed out while starting (AbortError). Click Restart Camera to retry.";
    }
    if (name === "SecurityError") {
      return "Camera requires HTTPS or http://localhost (SecurityError).";
    }
    return "Unable to start camera (" + name + (msg ? ": " + msg : "") + ").";
  }

  async function startCamera() {
    if (STATE.starting) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("This browser or insecure HTTP context does not support navigator.mediaDevices.getUserMedia(). Use http://localhost:8501 in Microsoft Edge or upload images on the right.");
      setStatus("Unsupported", "error");
      return;
    }

    STATE.starting = true;
    showError("");
    setStatus("Starting camera...", "");
    stopStream(true);

    const chosenId = selectEl.value || STATE.deviceId || "";
    STATE.deviceId = chosenId;

    const primaryConstraints = {
      video: chosenId
        ? { deviceId: { exact: chosenId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    };
    const fallbackConstraints = {
      video: chosenId ? { deviceId: { exact: chosenId } } : true,
      audio: false,
    };

    let stream = null;
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia(primaryConstraints);
      } catch (firstErr) {
        if (
          firstErr &&
          (firstErr.name === "OverconstrainedError" ||
           firstErr.name === "AbortError" ||
           firstErr.name === "NotFoundError")
        ) {
          stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        } else {
          throw firstErr;
        }
      }

      STATE.stream = stream;
      const tracks = stream.getVideoTracks();
      if (tracks.length > 0) {
        const activeSettings = tracks[0].getSettings ? tracks[0].getSettings() : {};
        if (activeSettings.deviceId) {
          STATE.deviceId = activeSettings.deviceId;
        }
        tracks[0].onended = () => {
          if (STATE.stream === stream) {
            STATE.stream = null;
            showError("Camera stream ended unexpectedly. Click Start Camera to reconnect.");
            syncUi();
          }
        };
      }
      await refreshDevices();
      syncUi();
    } catch (err) {
      STATE.stream = null;
      showError(formatMediaError(err));
      syncUi();
    } finally {
      STATE.starting = false;
    }
  }

  function capturePhoto() {
    if (!isStreamLive(STATE.stream)) {
      showError("Camera is not active. Click Start Camera first.");
      syncUi();
      return;
    }
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    setTriggerValue("captured_photo", dataUrl);
  }

  if (!STATE.beforeUnloadBound) {
    STATE.beforeUnloadBound = true;
    window.addEventListener("beforeunload", () => stopStream(true));
  }

  selectEl.onchange = () => {
    STATE.deviceId = selectEl.value || "";
    if (isStreamLive(STATE.stream)) {
      startCamera();
    }
  };
  btnStart.onclick = () => startCamera();
  btnRestart.onclick = () => startCamera();
  btnStop.onclick = () => stopStream(false);
  btnCapture.onclick = () => capturePhoto();

  refreshDevices().then(syncUi);
}
"""

_KX_CAMERA_RENDERER = st.components.v2.component(
    "kx_persistent_camera",
    html=_CAMERA_HTML,
    css=_CAMERA_CSS,
    js=_CAMERA_JS,
    isolate_styles=True,
)


def render_persistent_camera(key: str = "kx_live_camera") -> bytes | None:
    """Render the persistent Edge-compatible camera widget and return captured JPEG bytes if taken."""
    result = _KX_CAMERA_RENDERER(key=key)
    if not result:
        return None
    data_url = result.get("captured_photo")
    if not data_url or not isinstance(data_url, str):
        return None
    return decode_captured_data_url(data_url)


def decode_captured_data_url(data_url: str) -> bytes | None:
    """Decode a canvas data:image/...;base64,... URL into raw image bytes."""
    if not data_url or "," not in data_url:
        return None
    header, b64_payload = data_url.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        return None
    try:
        raw = base64.b64decode(b64_payload, validate=True)
        return raw if raw else None
    except (binascii.Error, ValueError):
        return None
