# 🎙️ InterviewAI — Resume-Aware Voice Practice

AI-powered mock interviews with resume-based questions, voice transcription, and body language analysis.  
Works on **any device** (desktop, mobile, tablet) via Streamlit Community Cloud.

---

## ⚡ Quick Deploy (Step-by-Step)

### Step 1 — Get your API keys

**Groq API (free):**
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Click **Create API Key** → copy it

**Metered.ca TURN server (free — needed for webcam on mobile/other devices):**
1. Go to [app.metered.ca/signup](https://app.metered.ca/signup) → sign up free
2. After login → click **TURN** in the left sidebar
3. You'll see your **App Name** (e.g. `myapp123`) and **API Key**
4. Copy both — you'll need them in Step 4

---

### Step 2 — Create a new GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. Name it `interviewai` (or anything you like)
3. Set to **Public**
4. Click **Create repository**

---

### Step 3 — Upload these files to GitHub

Upload ALL of the following files to your new repo:
```
app.py
webcam_component.py
requirements.txt
.streamlit/config.toml
README.md
.gitignore
```

**Do NOT upload:** `.env`, `yolov8n.pt` (too large), `.env.example`

To upload on GitHub.com:
- Click **Add file** → **Upload files**
- Drag all files in → click **Commit changes**

For `.streamlit/config.toml` — you need to create the folder first:
- Click **Add file** → **Create new file**
- Type `.streamlit/config.toml` as the filename
- Paste the contents from the config.toml file

---

### Step 4 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. Click **New app**
3. Select your repo → branch: `main` → Main file: `app.py`
4. Click **Advanced settings** → **Secrets**
5. Paste this (fill in YOUR values):

```toml
GROQ_API_KEY = "gsk_your_actual_groq_key"
METERED_API_KEY = "your_metered_api_key"
METERED_APP_NAME = "your_metered_app_name"
```

6. Click **Deploy** — takes ~3 minutes to build

---

### Step 5 — Test from your phone

Once deployed, open the Streamlit URL on your phone.
- The webcam should now connect properly via TURN servers
- Voice recording works in mobile Chrome and Safari

---

## 🔧 Run locally

```bash
# Clone your repo
git clone https://github.com/yourusername/interviewai
cd interviewai

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your keys

# Run
streamlit run app.py
```

---

## 📱 Mobile browser compatibility

| Feature | Chrome Android | Safari iOS | Samsung Browser |
|---------|---------------|-----------|-----------------|
| Voice recording | ✅ | ✅ | ✅ |
| Webcam (with TURN) | ✅ | ✅ | ✅ |
| PDF download | ✅ | ✅ | ✅ |
| TTS (question read aloud) | ✅ | ✅ | ✅ |

> **Note:** On iOS Safari, tap **Allow** when prompted for microphone and camera access.

---

## 🗂️ File structure

```
interviewai/
├── app.py                    ← Main Streamlit app
├── webcam_component.py       ← Body language analysis + TURN support
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Streamlit settings
├── .gitignore                ← Keeps secrets out of GitHub
├── .env.example              ← Template for local .env
└── README.md                 ← This file
```

---

## 🔑 Environment variables

| Variable | Where to get it | Required |
|----------|----------------|----------|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | ✅ Yes |
| `METERED_API_KEY` | [app.metered.ca](https://app.metered.ca) → TURN | For cross-device webcam |
| `METERED_APP_NAME` | [app.metered.ca](https://app.metered.ca) → your app name | For cross-device webcam |

---

## ❓ Troubleshooting

**Webcam not connecting on mobile:**
- Check your Metered.ca keys are correct in Streamlit Secrets
- Try a different browser (Chrome works best on Android, Safari on iOS)
- Make sure you clicked "Allow" for camera permissions

**Voice not transcribing:**
- Make sure you clicked Stop Recording before submitting
- Check your Groq API key is valid at console.groq.com

**App crashes on startup:**
- Check Streamlit logs (click "Manage app" → "Logs")
- Usually a missing package — check requirements.txt matches exactly

**yolov8n.pt model:**
- Ultralytics will auto-download it on first run (~6MB)
- This is normal — it only downloads once
