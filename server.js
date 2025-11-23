import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import axios from "axios";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  classifyLetterFromBase64,
  stopClassifier,
} from "./letterClassifier.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Store for accumulating letters per session
const letterSessions = new Map();

// Configuration for quality filtering
const MIN_CONFIDENCE =  0.7; // Only accept predictions above 70% confidence
const STABILITY_COUNT = parseInt(process.env.STABILITY_COUNT) || 3; // Require same letter 3 times in a row

function normalizeLetter(letter) {
  if (!letter || typeof letter !== "string") {
    return null;
  }

  const upper = letter.trim().toUpperCase();
  if (!/^[A-Z]$/.test(upper)) {
    return null;
  }

  return upper;
}

function accumulateLetter(letter, confidence, sessionId = "default") {
  if (!letterSessions.has(sessionId)) {
    letterSessions.set(sessionId, {
      letters: [],
      recentPredictions: [], // Track recent predictions for stability check
      updatedAt: Date.now(),
    });
  }

  const session = letterSessions.get(sessionId);
  const cleanLetter = normalizeLetter(letter);

  // Reject if confidence is too low
  if (confidence < MIN_CONFIDENCE) {
    return {
      sessionId,
      letters: session.letters,
      accumulatedText: session.letters.join(""),
      accepted: false,
      reason: "low_confidence",
    };
  }

  // Reject if letter is invalid
  if (!cleanLetter) {
    return {
      sessionId,
      letters: session.letters,
      accumulatedText: session.letters.join(""),
      accepted: false,
      reason: "invalid_letter",
    };
  }

  // Add to recent predictions (keep last STABILITY_COUNT predictions)
  session.recentPredictions.push(cleanLetter);
  if (session.recentPredictions.length > STABILITY_COUNT) {
    session.recentPredictions.shift();
  }

  // Check if we have enough stable predictions
  if (session.recentPredictions.length < STABILITY_COUNT) {
    return {
      sessionId,
      letters: session.letters,
      accumulatedText: session.letters.join(""),
      accepted: false,
      reason: "insufficient_stability",
      stabilityProgress: session.recentPredictions.length,
      stabilityRequired: STABILITY_COUNT,
    };
  }

  // Check if all recent predictions are the same (stability check)
  const allSame = session.recentPredictions.every((p) => p === cleanLetter);

  if (!allSame) {
    return {
      sessionId,
      letters: session.letters,
      accumulatedText: session.letters.join(""),
      accepted: false,
      reason: "unstable",
    };
  }

  // Only add if it's different from the last accumulated letter
  const lastLetter = session.letters[session.letters.length - 1];
  if (lastLetter !== cleanLetter) {
    session.letters.push(cleanLetter);
    session.updatedAt = Date.now();
    // Clear recent predictions after successful accumulation
    session.recentPredictions = [];
  }

  return {
    sessionId,
    letters: session.letters,
    accumulatedText: session.letters.join(""),
    accepted: true,
    acceptedLetter: cleanLetter,
  };
}

app.post("/api/classify-letter", async (req, res) => {
  try {
    const { image, sessionId = "default" } = req.body;

    if (!image) {
      return res.status(400).json({
        error: "No image provided. Send a base64 encoded frame.",
      });
    }

    const classification = await classifyLetterFromBase64(image);
    const accumulation = accumulateLetter(
      classification.letter,
      classification.confidence,
      sessionId
    );

    res.json({
      letter: classification.letter,
      confidence: classification.confidence,
      accumulatedText: accumulation.accumulatedText,
      letters: accumulation.letters,
      sessionId: accumulation.sessionId,
      accepted: accumulation.accepted || false,
      acceptedLetter: accumulation.acceptedLetter || null,
      reason: accumulation.reason || null,
      stabilityProgress: accumulation.stabilityProgress || null,
      stabilityRequired: accumulation.stabilityRequired || null,
    });
  } catch (error) {
    console.error("Error in /api/classify-letter:", error);
    res.status(500).json({
      error: "Failed to classify letter",
      message: error.message,
    });
  }
});

app.post("/api/send-to-gemini", async (req, res) => {
  try {
    const { sessionId = "default", text } = req.body || {};
    const session = letterSessions.get(sessionId);
    const accumulatedText = text?.trim() || session?.letters?.join("");

    if (!accumulatedText) {
      return res
        .status(400)
        .json({ error: "No accumulated text found for this session." });
    }

    const sanitized = accumulatedText.trim();
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      systemInstruction:
        "You turn raw ASL letter transcripts into natural sentences. " +
        "Return a single concise sentence. Do not add explanations, filler words, " +
        "or commentary. Output only the sentence.",
    });

    const prompt = `Letters captured from ASL fingerspelling: "${sanitized}". Convert them into a natural English sentence. Respond with only the final sentence.`;
    const result = await model.generateContent(prompt);
    const sentence = result.response.text().trim();

    if (letterSessions.has(sessionId)) {
      letterSessions.delete(sessionId);
    }

    res.json({
      sentence,
      originalText: sanitized,
    });
  } catch (error) {
    console.error("Error in /api/send-to-gemini:", error);
    res.status(500).json({
      error: "Failed to generate sentence",
      message: error.message,
    });
  }
});

// POST /text-to-speech endpoint - converts text to speech using ElevenLabs
app.post("/text-to-speech", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    const elevenLabsApiKey = process.env.ELEVENLABS_API_KEY;
    if (!elevenLabsApiKey) {
      return res
        .status(500)
        .json({ error: "ElevenLabs API key not configured" });
    }

    const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";

    const ttsResponse = await axios.post(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        text: text,
        model_id: "eleven_turbo_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.5,
        },
      },
      {
        headers: {
          Accept: "audio/mpeg",
          "Content-Type": "application/json",
          "xi-api-key": elevenLabsApiKey,
        },
        responseType: "arraybuffer",
      }
    );

    // Save audio file
    const audioDir = path.join(__dirname, "public", "audio");
    if (!fs.existsSync(audioDir)) {
      fs.mkdirSync(audioDir, { recursive: true });
    }

    const audioFileName = `audio_${Date.now()}.mp3`;
    const audioPath = path.join(audioDir, audioFileName);
    fs.writeFileSync(audioPath, ttsResponse.data);

    const audioUrl = `/audio/${audioFileName}`;

    res.json({
      audioUrl: audioUrl,
      text: text,
    });
  } catch (error) {
    console.error("Error in /text-to-speech endpoint:", error);

    if (error.response) {
      const status = error.response.status;
      let errorMessage = error.message;

      try {
        if (error.response.data instanceof Buffer) {
          const errorText = error.response.data.toString("utf-8");
          const errorJson = JSON.parse(errorText);
          errorMessage =
            errorJson.detail?.message || errorJson.detail?.status || errorText;
        } else if (typeof error.response.data === "object") {
          errorMessage =
            error.response.data.detail?.message ||
            error.response.data.detail?.status ||
            JSON.stringify(error.response.data);
        }
      } catch (parseError) {
        // If parsing fails, use the original message
      }

      return res.status(status).json({
        error: "ElevenLabs API error",
        message: errorMessage,
        status: status,
      });
    }

    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

// POST /process-complete endpoint - complete flow: recognize -> accumulate -> form sentence -> TTS
// POST /clear-session endpoint - clears accumulated letters for a session
app.post("/api/clear-session", (req, res) => {
  const { sessionId = "default" } = req.body || {};

  if (letterSessions.has(sessionId)) {
    letterSessions.delete(sessionId);
  }

  res.json({ success: true, sessionId });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

const shutdown = () => {
  stopClassifier();
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("exit", shutdown);
