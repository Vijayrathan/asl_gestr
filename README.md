# Gestr · ASL Fingerspelling Assistant

Gestr turns live ASL fingerspelling into natural English sentences. The workflow is simple:

1. Capture live webcam video in the browser.
2. Send frames to a TensorFlow-based Python worker that predicts ASL letters (A–Z).
3. Accumulate the letters per session.
4. Send the accumulated raw letters to Gemini, which responds with **only** the final sentence—no filler, no commentary.
5. (Optional) Push the sentence through the ElevenLabs TTS endpoint for audio playback.

## Features

- 🎥 **Live Camera Capture** – Start/stop camera directly from the UI.
- 🔤 **Letter-Level Recognition** – TensorFlow model predicts individual ASL letters.
- ✨ **Gemini Polishing** – Gemini receives the exact accumulated letters and returns a single polished sentence.
- 🧹 **Session Control** – Clear or resend whenever you need.
- 🔊 **TTS Ready** – Keep the ElevenLabs endpoint handy if you want audio output.

## Setup

### 1. Install Node.js dependencies

```bash
npm install
```

### 2. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

Python powers the TensorFlow classifier (`classifier_worker.py`) that stays alive while the Node server runs.

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
PORT=3000
GEMINI_API_KEY=your_gemini_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # optional
```

### 4. Start the server

```bash
npm start
```

For development with file watching:

```bash
npm run dev
```

### 5. Open the app

Visit `http://localhost:3000` and grant camera access.

## Usage

1. **Start Camera** – click “Start Camera” to open your webcam.
2. **Fingerspell** – make each letter in front of the camera; the app samples frames automatically.
3. **Watch Text Build** – latest letter + accumulated text update in real time.
4. **Send** – click “Send to Gemini” once you finish spelling; Gemini returns a sentence and nothing else.
5. **Clear / Repeat** – “Clear” resets the current session; “Stop Camera” halts the video stream.

## API Endpoints

### POST `/api/classify-letter`

Classify a single frame and accumulate the resulting letter.

**Request**

```json
{
  "image": "data:image/jpeg;base64,...",
  "sessionId": "session_123"
}
```

**Response**

```json
{
  "letter": "H",
  "confidence": 0.94,
  "accumulatedText": "HEL",
  "letters": ["H", "E", "L"],
  "sessionId": "session_123"
}
```

### POST `/api/send-to-gemini`

Send the accumulated letters (or explicit text) to Gemini. The response is always a single sentence.

**Request**

```json
{
  "sessionId": "session_123"
}
```

**Response**

```json
{
  "sentence": "Hello, I need help.",
  "originalText": "HELLOINEEDHELP"
}
```

### POST `/api/clear-session`

Clear cached letters for a session.

```json
{
  "sessionId": "session_123"
}
```

**Response**

```json
{
  "success": true,
  "sessionId": "session_123"
}
```

### POST `/text-to-speech`

Optional endpoint that relays text to ElevenLabs and stores the returned audio.

```json
{
  "text": "Hello, how are you?"
}
```

## Architecture

- **Frontend**: Vanilla HTML/CSS/JS served from `public/`. A hidden canvas grabs video frames before sending them to the backend.
- **Backend**: Express.js (`server.js`) provides the REST API and keeps session state.
- **Letter Classification**: `letterClassifier.js` spawns `classifier_worker.py`, keeping the TensorFlow model warm and communicating over stdin/stdout.
- **Sentence Formation**: Gemini Flash with strict instructions to reply with the sentence only.
- **Text-to-Speech**: ElevenLabs endpoint (optional, unchanged).

## File Structure

```
gestr/
├── server.js              # Express server + routes
├── letterClassifier.js    # Node bridge to the Python worker
├── classifier_worker.py   # TensorFlow classifier (stdin/stdout loop)
├── inference.py           # Standalone webcam demo for debugging
├── asl_model.keras        # Keras model weights
├── asl_class_map.npy      # Class index → letter mapping
├── public/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt       # Python deps (TensorFlow, OpenCV, NumPy)
├── package.json
└── README.md
```

## Improving Recognition

- Retrain or fine-tune `asl_model.keras` with more diverse data.
- Update `asl_class_map.npy` if the class ordering changes.
- Add temporal smoothing or majority voting inside `accumulateLetter` if you want stricter filtering.

## Troubleshooting

**Camera issues**

- Confirm browser permissions are granted.
- Chrome/Safari may require HTTPS for camera access; localhost works without it.
- Open DevTools for detailed console errors.

**Python worker issues**

- Ensure Python 3.9+ is installed: `python3 --version`.
- Reinstall dependencies: `pip3 install -r requirements.txt`.
- Watch the Node logs—worker stderr is piped there for easier debugging.

**API errors**

- Verify `.env` values, especially `GEMINI_API_KEY`.
- Make sure the server has internet access for Gemini/ElevenLabs calls.
- Use the Network tab in DevTools to inspect payloads/responses.

## License

ISC
