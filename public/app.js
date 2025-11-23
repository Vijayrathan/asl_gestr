const videoEl = document.getElementById("camera");
const canvasEl = document.getElementById("frame-canvas");
const ctx = canvasEl.getContext("2d");

const latestLetterEl = document.getElementById("latest-letter");
const confidenceEl = document.getElementById("confidence");
const accumulatedTextEl = document.getElementById("accumulated-text");
const geminiOutputEl = document.getElementById("gemini-output");
const statusLogEl = document.getElementById("status-log");

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

let mediaStream = null;
let captureTimer = null;
let isProcessingFrame = false;
let accumulatedText = "";
const sessionId = crypto.randomUUID();

const CAPTURE_INTERVAL_MS = 1100;

function updateStatus(message) {
  const timestamp = new Date().toLocaleTimeString();
  statusLogEl.textContent = `[${timestamp}] ${message}`;
}

function setButtonsState({ capturing }) {
  startBtn.disabled = capturing;
  stopBtn.disabled = !capturing;
  sendBtn.disabled = !accumulatedText;
  clearBtn.disabled = !capturing && !accumulatedText;
}

async function startCamera() {
  if (mediaStream) {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 640, height: 480 },
      audio: false,
    });
    videoEl.srcObject = mediaStream;
    await videoEl.play();
    updateStatus("Camera started. Begin fingerspelling letters.");
    setButtonsState({ capturing: true });
    scheduleCapture();
  } catch (error) {
    console.error("Camera error:", error);
    updateStatus(`Unable to access camera: ${error.message}`);
  }
}

function stopCamera() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  videoEl.srcObject = null;
  updateStatus("Camera stopped.");
  setButtonsState({ capturing: false });
}

function resetSessionState() {
  accumulatedText = "";
  latestLetterEl.textContent = "--";
  confidenceEl.textContent = "--";
  accumulatedTextEl.textContent = "Nothing yet";
  geminiOutputEl.textContent = "Waiting for send...";
}

function scheduleCapture() {
  captureTimer = setInterval(() => {
    if (!mediaStream || isProcessingFrame) {
      return;
    }
    captureAndSendFrame();
  }, CAPTURE_INTERVAL_MS);
}

function drawFrameToCanvas() {
  const { videoWidth, videoHeight } = videoEl;
  if (!videoWidth || !videoHeight) {
    return null;
  }
  canvasEl.width = videoWidth;
  canvasEl.height = videoHeight;
  ctx.drawImage(videoEl, 0, 0, videoWidth, videoHeight);
  return canvasEl.toDataURL("image/jpeg", 0.8);
}

async function captureAndSendFrame() {
  const dataUrl = drawFrameToCanvas();
  if (!dataUrl) {
    return;
  }

  isProcessingFrame = true;

  try {
    const response = await fetch("/api/classify-letter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: dataUrl,
        sessionId,
      }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || "Classification failed");
    }

    const payload = await response.json();

    // Only update the displayed letter if it was accepted and added to accumulation
    if (payload.accepted && payload.acceptedLetter) {
      latestLetterEl.textContent = payload.acceptedLetter;
      confidenceEl.textContent = payload.confidence
        ? `${(payload.confidence * 100).toFixed(1)}%`
        : "--";
    }
    // If not accepted, don't update the latest letter display
    // This prevents random oscillations from showing up
    // The letter will only update when a letter passes all checks

    // Always update accumulated text (only contains accepted letters)
    accumulatedText = payload.accumulatedText || "";
    accumulatedTextEl.textContent = accumulatedText || "Nothing yet";
    setButtonsState({ capturing: Boolean(mediaStream) });
  } catch (error) {
    console.error("Classification error:", error);
    updateStatus(`Classification failed: ${error.message}`);
  } finally {
    isProcessingFrame = false;
  }
}

async function sendToGemini() {
  if (!accumulatedText) {
    updateStatus("No letters captured yet.");
    return;
  }

  try {
    sendBtn.disabled = true;
    updateStatus("Sending accumulated text to Gemini...");

    const response = await fetch("/api/send-to-gemini", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || "Gemini request failed");
    }

    const payload = await response.json();
    geminiOutputEl.textContent = payload.sentence || "No sentence returned.";
    updateStatus("Gemini generated a sentence.");
    accumulatedText = "";
    accumulatedTextEl.textContent = "Cleared after send.";
    setButtonsState({ capturing: Boolean(mediaStream) });
  } catch (error) {
    console.error("Gemini error:", error);
    updateStatus(`Gemini request failed: ${error.message}`);
    sendBtn.disabled = false;
  }
}

async function clearSession() {
  try {
    await fetch("/api/clear-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });
  } catch (error) {
    console.warn("Failed to clear session on server:", error);
  } finally {
    resetSessionState();
    updateStatus("Session cleared.");
    setButtonsState({ capturing: Boolean(mediaStream) });
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
sendBtn.addEventListener("click", sendToGemini);
clearBtn.addEventListener("click", clearSession);

setButtonsState({ capturing: false });

