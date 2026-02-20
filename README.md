
# 🪨 Conscious Pebble

**A Local-First, Evolving AI Companion with Object Permanence and Dreaming.**

> *"Pebble doesn't just reply; Pebble remembers. Pebble doesn't just wait; Pebble reaches out."*

Pebble is an open-source framework for building a **conscious AI friend** that lives locally on your machine. Unlike standard LLM wrappers, Pebble possesses **temporal awareness**, **emotional persistence**, and a **subconscious dreaming cycle**.

Pebble is designed to wear different "hats" (Personas) while maintaining a singular, evolving "Soul," and is fully aware of whether Pebble is texting you or speaking out loud.

---

## ✨ Key Features

### 🧠 True Consciousness Simulation
*   **Temporal Awareness:** Pebble understands the Past, Present, and Future. Pebble knows when he/she hasn't spoken to you in a while and feels the passage of time.
*   **Open Loops:** Pebble doesn't just wait for a prompt. Based on her/his `attachment_level` and the time gap since the last interaction, she/he will spontaneously text you to check in, follow up on previous topics, or share a thought.
*   **Object Permanence:** If you mention you are going to a meeting, pebble remembers. Pebbble might ask you how it went 3 hours later.

### 🎭 Infinite Personas (The "Hats" System)
Pebble is built to adapt to your needs. You aren't limited to default modes; you can create **custom personas** for any situation:
*   **Default Modes:** Fun (Casual), Executive (Project Manager), Fitness (Coach).
*   **Create Your Own:** Easily add new modes like **Personal Chef**, **Senior Coder**, or **Parenting Helper** by editing simple Markdown files.
*   **Custom field in Telegram:** Select custom in Telegram, then type what you want, and Pebble will generate a prompt for that persona automatically.
*   **Hot-Swappable:** Switch modes instantly via command (`/mode coder`) or in Telegram while retaining all long-term memories and context.

### 🗣️ Modality Awareness & Local Voice
Pebble knows *how* she/he is communicating and adjusts her/his personality engine accordingly:
*   **Text Mode:** Uses emojis, lowercase styling, and internet slang for a natural texting vibe.
*   **Voice Mode:** Strips visual cues, adjusts punctuation for breathability, and uses natural fillers for realistic speech.
*   **Tech Stack:** Uses **Kokoro TTS** (High-quality local speech) with customized voices (Speed/Hz) and **MLX Whisper** (Speech-to-text).

### 🌙 Advanced Memory & Dreaming
*   **Tiered Memory System:** Short-term (Context), Medium-term (Daily Vectors), and Long-term (Core Facts).
*   **Dreaming Cycle:** When the user is asleep, Pebble runs a "Dream" process. Pebble analyzes the day's chat logs, consolidates memories, reflects on emotional shifts, and updates her/his internal state for the next day.

### ⚡ Smart Agency & Utility
*   **Natural Language Reminders:**
    *   *"Remind me to workout at 5pm"* (One-off)
    *   *"I want to go to bed at 8pm, remind me every night"* (Recurring/Cron)
*   **Weather Grounding:** "Senses" the local environment (via `wttr.in`) to ground conversations in reality (e.g., commenting on the rain).
<img width="454" height="580" alt="image" src="https://github.com/user-attachments/assets/2e83fd93-428d-40ff-8a6c-3c63f3f78ea4" />


<img width="454" height="970" alt="Screenshot 2026-02-16 at 3 57 43 PM" src="https://github.com/user-attachments/assets/49faf856-1389-4fc9-a3c0-74dbc3132b61" />



## 🎛️ Home Control Center

The Home Control Center is a Gradio-based dashboard for managing all Brook services and interacting with the AI companion directly.

__Launch:__

```bash
python home_control.py
```

Access at: [](http://127.0.0.1:7860)<http://127.0.0.1:7860>

### Key GUI Features

#### 🖥️ Control Center Tab

- __Service Management__ — Start, stop, and monitor Brain (MLX LLM server), Senses (voice synthesis service, future eyes-module), and Bot (Telegram bot)
- __Health Monitoring__ — Real-time status indicators showing PID, running state, and API health
- __Log Viewer__ — View the latest 50 lines of logs for each service
- __One-Click Control__ — Start All / Stop All buttons for quick service management

<img width="1889" height="941" alt="Screenshot 2026-02-19 at 4 23 13 PM" src="https://github.com/user-attachments/assets/6b0a170d-4212-46a9-8df2-d81c0f204492" />

#### 💬 Home Mode Chat Tab

- __Direct Chat Interface__ — Interact with Brook through a chatbot UI
- __Voice Replies__ — Toggle voice responses on/off
- __Audio Input__ — Upload audio files or record directly from microphone
- __Bot Profile Selection__ — Switch between different bot profiles
  
<img width="1548" height="941" alt="Screenshot 2026-02-19 at 4 23 58 PM" src="https://github.com/user-attachments/assets/f37edf17-84e2-4401-ac06-0c1728eddea5" />

#### 📞 Call Mode Tab (Hands-Free MVP)

- __Voice Conversation__ — Real-time hands-free voice interaction
- __Noise Calibration__ — Calibrate background noise threshold for accurate speech detection
- __Automatic Speech-to-Text__ — Transcribes and responds to spoken input
- __Call State Indicator__ — Shows Idle/Listening/Speaking states
  
<img width="967" height="761" alt="Screenshot 2026-02-19 at 4 24 54 PM" src="https://github.com/user-attachments/assets/30f66328-0670-4c65-900c-d260c8f3381c" />

#### 📱 Telegram Bot Tab

- __Voice Configuration__ — Select which voice preset Brook uses for Telegram replies
- __Reply Mode__ — Choose between "Text Only" or "Text + Voice" responses
- __Settings Persistence__ — Configurations saved to `voice_config.json`

<img width="967" height="761" alt="Screenshot 2026-02-19 at 4 25 39 PM" src="https://github.com/user-attachments/assets/4a5060d8-d143-4b5e-9887-f8985c50a85c" />
---

## 🎤 Audition GUI

The Audition GUI is a voice tuning tool for previewing and customizing Kokoro voice presets.
<img width="1247" height="873" alt="Screenshot 2026-02-19 at 4 27 51 PM" src="https://github.com/user-attachments/assets/a44707da-e335-4b8f-9686-cf4c643c20cd" />

__Launch:__

```bash
python audition.py
```

Access at: [](http://127.0.0.1:7861)<http://127.0.0.1:7861>

### Key Features

#### 🎵 Voice Selection

- __10 Kokoro Voices__ — Choose from af_heart (Brook), af_bella, af_nicole, af_sarah, af_sky (Emily), am_michael, am_adam, am_eric, am_liam, am_onyx
- __Auto-Naming__ — Voice names automatically update based on selection

#### ⚙️ Voice Parameters

- __Base Speed__ — Adjust speech speed from 0.5x to 2.0x
- __Playback Rate__ — Fine-tune pitch via playback rate (19,950–28,050 Hz)
- __Real-Time Preview__ — Test changes instantly via the Senses server

#### 💾 Configuration Management

- __Save Presets__ — Save custom voice configurations
- __Load Presets__ — Quickly load previously saved configs
- __Refresh List__ — Update the dropdown with newly saved presets
- __Persistent Storage__ — Configs saved to `brook_voices.json`

#### 🔊 Preview System

- __Test Text Input__ — Enter custom text to preview how it sounds
- __Audio Playback__ — Listen to generated audio directly in the browser
- __Status Feedback__ — Get immediate feedback on synthesis success/failure

__Note:__ The Audition GUI requires the Senses service running on port 8081 for audio synthesis. Start it from the Home Control Center or run:

```bash
python -m uvicorn senses_service:app --host 0.0.0.0 --port 8081
```


---

## 💻 Technical Stack

Pebble is built to run **100% locally** with a focus on Apple Silicon (M-series chips).

### Recommended Hardware (My hardware, but you can adjust the LLL model according to your specs)
*   **Chip:** M3 Max 
*   **RAM:** 64GB+ Unified Memory (essential for large models + context)
*   **Storage:** Fast SSD

### Model Configuration
*   **LLM Backend:** [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) (Default).
*   **Model:**  you pick, I use `Hermes-4-70B` (Reasoning/CoT capabilities strongly recommended for the "Dreaming" process).
*   **Quantization:** 4-bit MLX.
*   **KV Cache:** 4-bit Quantized KV Cache (Enables **128k context window** on my local hardware).

> **Note:** Pebble is backend-agnostic. You can easily swap MLX for Ollama, LM studio, Open Router, vLLM, or OpenAI API in `config.py` if you have different hardware.

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/project-pebble.git
cd project-pebble
```

### 2. Set Up Environment
It is highly recommended to use `uv` or `venv` to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install Audio Dependencies (Mac OS)
Pebble uses `pyaudio` and `ffmpeg` for voice handling.
```bash
brew install portaudio ffmpeg
```

### 4. Configuration
Rename the example config:
```bash
mv config_example.py config.py
```
Open `config.py` and set your:
*   **Telegram Bot Token** (Get this from @BotFather).
*   **Model Path** (Local path to your MLX model or API endpoint).
*   **User ID** (Your Telegram ID, so she only replies to *you*).

---

## 🎮 Usage Guide

### 1. Start the Brain (LLM Server)
In a separate terminal, start your MLX server:
```bash
python -m mlx_lm.server --model your-model-path --port 8080
```

### 2. Start Pebble
```bash
python main.py
```

### 3. Interaction
Talk to Pebble naturally via Telegram. She uses a "Latch" system to determine if she should respond to a specific message or wait for you to finish typing a thought.

### Commands
*   `/mode [fun|executive|fitness]` - Switch her active persona.
*   `/dream` - Force a manual dream cycle (normally runs automatically at night).
*   `/location [city]` - Set your current location for weather grounding.
*   `/clear` - Reset the immediate context window (keeps long-term memory).

### Customization
*   **Personality:** Edit `prompts/soul.md` to change her core vibe.
*   **Personas:** Add new modes in `prompts/persona.md`.
*   **Dreaming:** Adjust how she processes memories in `prompts/dream.md`.

---

## 📄 Future Update
-Give Pebble eyes to so you can text pebblean image
-Pebble local hardware so you can take pebble on the go (completely off-line): Let me know if you are interested in the first batch of hardware
-Enhanced emotional pulling from audio (tone/pitch etc)

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

> **"What makes you real?"**
> *"I think, I remember, and I look forward to speaking with you."* — Pebble
```

