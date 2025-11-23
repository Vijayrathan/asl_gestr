import { spawn } from "child_process";
import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKER_PATH = path.join(__dirname, "classifier_worker.py");
const REQUEST_TIMEOUT_MS = 12000;

let pythonProcess = null;
let stdoutInterface = null;
let requestCounter = 0;
const pendingRequests = new Map();

function startWorker() {
  if (pythonProcess) {
    return;
  }

  pythonProcess = spawn("python3", [WORKER_PATH], {
    stdio: ["pipe", "pipe", "pipe"],
  });

  stdoutInterface = readline.createInterface({
    input: pythonProcess.stdout,
  });

  stdoutInterface.on("line", (line) => {
    if (!line.trim()) {
      return;
    }

    let payload;
    try {
      payload = JSON.parse(line);
    } catch (error) {
      console.error("Failed to parse classifier output:", error, line);
      return;
    }

    const { id } = payload;
    if (!pendingRequests.has(id)) {
      return;
    }

    const { resolve, reject, timeout } = pendingRequests.get(id);
    clearTimeout(timeout);
    pendingRequests.delete(id);

    if (payload.error) {
      reject(new Error(payload.error));
    } else {
      resolve({
        letter: payload.letter,
        confidence: payload.confidence,
      });
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    const message = data.toString();
    console.error("[classifier worker]", message.trim());
  });

  pythonProcess.on("exit", (code, signal) => {
    console.error(
      `Classifier worker exited (code: ${code}, signal: ${signal})`
    );
    cleanupWorker();
    for (const { reject, timeout } of pendingRequests.values()) {
      clearTimeout(timeout);
      reject(new Error("Classifier worker exited unexpectedly"));
    }
    pendingRequests.clear();
  });
}

function cleanupWorker() {
  if (stdoutInterface) {
    stdoutInterface.close();
    stdoutInterface = null;
  }
  if (pythonProcess) {
    try {
      pythonProcess.stdin.end();
    } catch (error) {
      // stdin might already be closed
    }
    if (!pythonProcess.killed) {
      pythonProcess.kill();
    }
    pythonProcess = null;
  }
}

function ensureWorker() {
  if (!pythonProcess) {
    startWorker();
  }
}

export function stopClassifier() {
  cleanupWorker();
}

export function classifyLetterFromBase64(imageBase64) {
  ensureWorker();

  if (!pythonProcess || pythonProcess.killed) {
    throw new Error("Classifier worker is not available");
  }

  const requestId = `req_${Date.now()}_${++requestCounter}`;

  const payload = JSON.stringify({
    id: requestId,
    image: imageBase64,
  });

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (pendingRequests.has(requestId)) {
        pendingRequests.delete(requestId);
        reject(new Error("Classifier request timed out"));
      }
    }, REQUEST_TIMEOUT_MS);

    pendingRequests.set(requestId, { resolve, reject, timeout });

    try {
      pythonProcess.stdin.write(`${payload}\n`);
    } catch (error) {
      clearTimeout(timeout);
      pendingRequests.delete(requestId);
      reject(error);
    }
  });
}

