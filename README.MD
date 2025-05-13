# Healthcare Booking Voice Assistant

This project implements a **voice-enabled healthcare appointment booking agent**. It listens to the user's speech, extracts required information using an LLM, validates and stores patient details, and confirms a medical appointment via email.

Built using [`livekit-agents`](https://github.com/livekit/agents) with integrations for voice transcription (Deepgram), text-to-speech (Cartesia), LLM processing (OpenAI), and address validation (Smarty).

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file with all required keys. A sample configuration is provided in `.env.example`:

```bash
cp .env.example .env
```

You'll need valid credentials for:

* **Smarty** (Address validation API)
* **Zapier Webhook URL** (for email confirmation)

---

## Running the Agent

Start the voice assistant in a LiveKit room:

```bash
python main.py download-files
python main.py dev
```
