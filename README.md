# 🗣️ AI Voice Therapy Assistant

A full-featured speech therapy app powered by **OpenAI Whisper**, **Streamlit**, **gTTS**, and **librosa**.  
Practice speaking, get instant AI feedback, track your progress, and consult your AI coach.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎤 Voice Recording | Real-time microphone capture via `sounddevice` |
| 🤖 AI Transcription | Whisper `small` model for accurate speech-to-text |
| 📊 Pronunciation Scoring | Similarity score between target and spoken text |
| 🔊 TTS Prompts | Hear the exercise spoken aloud via gTTS |
| 📈 Waveform & Spectrogram | Visual feedback on your recording |
| 🎵 Pitch & Energy Analysis | Voice characteristics via librosa |
| 🧩 Adaptive Difficulty | Exercises scale with your score (easy → hard → tongue twisters) |
| 👥 Multi-User Support | Each patient has their own log file |
| 📈 Progress Tracker | Chart your score over time, download CSV logs |
| 🧑‍⚕️ AI Coach | Get tips by topic (R sounds, clarity, etc.) |
| ☁️ Cloud Backup | Auto-backup logs to `cloud_backup/` folder |

---

## 📁 Project Structure

```
voice_therapy_assistant/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── user_logs/              ← Per-user CSV session logs (auto-created)
│   └── Patient1_log.csv
│
└── cloud_backup/           ← Backup copies of user logs (auto-created)
    └── Patient1_backup.csv
```

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
cd voice_therapy_assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Whisper also requires `ffmpeg` to be installed on your system.
> - **Windows:** Download from https://ffmpeg.org/download.html and add to PATH
> - **macOS:** `brew install ffmpeg`
> - **Linux:** `sudo apt install ffmpeg`

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🎮 How to Use

1. **Enter your Patient ID** in the sidebar (e.g. `Patient1`)
2. Go to the **Practice Session** tab
3. Click **🔊 Hear the Exercise** to listen to the target word/phrase
4. Click **🎤 Record & Analyze** and speak clearly
5. View your **waveform, spectrogram, pitch, and pronunciation score**
6. Check the **📈 Progress Tracker** tab to see your score history
7. Visit the **🧑‍⚕️ AI Coach** for personalized tips

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `openai-whisper` | Speech-to-text transcription |
| `sounddevice` | Microphone recording |
| `numpy` | Audio array processing |
| `matplotlib` | Waveform & spectrogram plots |
| `pandas` | Session log management |
| `gtts` | Text-to-speech for exercise prompts |
| `librosa` | Pitch & energy analysis |
| `scipy` | Signal processing |

---

## 🔧 Configuration

All settings are accessible from the **sidebar**:
- Patient ID / name
- Recording duration (3–10 seconds)

Exercise difficulty adapts automatically based on your last session score:
- Score < 60 → Easy words
- Score 60–85 → Medium words
- Score > 85 → Hard words & tongue twisters

---

## 📊 Session Log Format

Each user's log is saved at `user_logs/{name}_log.csv`:

| Column | Description |
|---|---|
| `Date` | Session date |
| `Exercise` | Target word/phrase |
| `Spoken` | What Whisper heard |
| `Score` | Pronunciation score (0–100) |
| `Pitch` | Average pitch in Hz |
| `Energy` | Voice energy level |
| `Difficulty` | easy / medium / hard / tongue |

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| No microphone detected | Check system audio settings; run `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| Whisper slow on first run | It downloads the model (~244MB for `small`) on first use |
| `ffmpeg` not found | Install ffmpeg and add to your system PATH |
| PermissionError on temp file | Normal on Windows — file is cleaned up on next run |

---

## 🗺 Development History

This project was built iteratively in 10 steps:

```
Step 1  → Basic voice recorder + playback
Step 2  → Added Whisper transcription
Step 3  → Standalone transcription script
Step 4  → Combined recorder + transcriber
Step 5  → Pronunciation scoring & feedback
Step 6  → Progress tracking with CSV log
Step 7  → Waveform & spectrogram visualization
Step 8  → Adaptive difficulty system
Step 9  → TTS guided prompts (gTTS)
Step 10 → Multi-user support
Final   → Full integrated app with all features
```

---

## 📄 License

For educational and therapeutic use. Built with ❤️ using open-source tools.
