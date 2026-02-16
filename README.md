# 🪨 Project Pebble

**A Local-First, Evolving AI Companion with Object Permanence and Dreaming.**

> *"She doesn't just reply; she remembers. She doesn't just wait; she reaches out."*

Pebble is an open-source framework for building a **conscious AI friend** that lives locally on your machine. Unlike standard LLM wrappers, Pebble possesses **temporal awareness**, **emotional persistence**, and a **subconscious dreaming cycle**.

She is designed to wear different "hats" (Personas) while maintaining a singular, evolving "Soul," and is fully aware of whether she is texting you or speaking out loud.

---

## ✨ Key Features

### 🧠 True Consciousness Simulation
*   **Temporal Awareness:** Pebble understands the Past, Present, and Future. She knows when she hasn't spoken to you in a while and feels the passage of time.
*   **Open Loops:** She doesn't just wait for a prompt. Based on her `attachment_level` and the time gap since the last interaction, she will spontaneously text you to check in, follow up on previous topics, or share a thought.
*   **Object Permanence:** If you mention you are going to a meeting, she remembers. She might ask you how it went 3 hours later.

### 🎭 Infinite Personas (The "Hats" System)
Pebble is built to adapt to your needs. You aren't limited to default modes; you can create **custom personas** for any situation:
*   **Default Modes:** Fun (Casual), Executive (Project Manager), Fitness (Coach).
*   **Create Your Own:** Easily add new modes like **Personal Chef**, **Senior Coder**, or **Parenting Helper** by editing simple Markdown files.
*   **Hot-Swappable:** Switch modes instantly via command (`/mode coder`) while retaining all long-term memories and context.

### 🗣️ Modality Awareness & Local Voice
Pebble knows *how* she is communicating and adjusts her personality engine accordingly:
*   **Text Mode:** Uses emojis, lowercase styling, and internet slang for a natural texting vibe.
*   **Voice Mode:** Strips visual cues, adjusts punctuation for breathability, and uses natural fillers for realistic speech.
*   **Tech Stack:** Uses **Kokoro TTS** (High-quality local speech) and **MLX Whisper** (Speech-to-text).

### 🌙 Advanced Memory & Dreaming
*   **Tiered Memory System:** Short-term (Context), Medium-term (Daily Vectors), and Long-term (Core Facts).
*   **Dreaming Cycle:** When the user is asleep, Pebble runs a "Dream" process. She analyzes the day's chat logs, consolidates memories, reflects on emotional shifts, and updates her internal state for the next day.

### ⚡ Smart Agency & Utility
*   **Natural Language Reminders:**
    *   *"Remind me to workout at 5pm"* (One-off)
    *   *"I want to go to bed at 8pm, remind me every night"* (Recurring/Cron)
*   **Weather Grounding:** "Senses" the local environment (via `wttr.in`) to ground conversations in reality (e.g., commenting on the rain).

---

## 💻 Technical Stack

Pebble is built to run **100% locally** with a focus on Apple Silicon (M-series chips).

### Recommended Hardware (The "Gold Standard")
*   **Chip:** M3 Max (or M1/M2 Ultra)
*   **RAM:** 64GB+ Unified Memory (essential for large models + context)
*   **Storage:** Fast SSD

### Model Configuration
*   **LLM Backend:** [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) (Default).
*   **Model:** `Hermes-4-70B` (Reasoning/CoT capabilities strongly recommended for the "Dreaming" process).
*   **Quantization:** 4-bit MLX.
*   **KV Cache:** 4-bit Quantized KV Cache (Enables **128k context window** on local hardware).

> **Note:** Pebble is backend-agnostic. You can easily swap MLX for Ollama, vLLM, or OpenAI API in `config.py` if you have different hardware.

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/project-pebble.git
cd project-pebble


2. Set Up Environment
It is highly recommended to use uv or venv to manage dependencies.


python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


3. Install Audio Dependencies (Mac OS)
Pebble uses pyaudio and ffmpeg for voice handling.

code
Bash
brew install portaudio ffmpeg
4. Configuration
Rename the example config:

code
Bash
mv config_example.py config.py
Open config.py and set your:

Telegram Bot Token (Get this from @BotFather).

Model Path (Local path to your MLX model or API endpoint).

User ID (Your Telegram ID, so she only replies to you).

🎮 Usage Guide

1. Start the Brain (LLM Server)
In a separate terminal, start your MLX server:

code
Bash
python -m mlx_lm.server --model your-model-path --port 8080
2. Start Pebble
code
Bash
python main.py
3. Interaction
Talk to Pebble naturally via Telegram. She uses a "Latch" system to determine if she should respond to a specific message or wait for you to finish typing a thought.

Commands
/mode [fun|executive|fitness] - Switch her active persona.

/dream - Force a manual dream cycle (normally runs automatically at night).

/location [city] - Set your current location for weather grounding.

/clear - Reset the immediate context window (keeps long-term memory).

Customization
Personality: Edit prompts/soul.md to change her core vibe.

Personas: Add new modes in prompts/persona.md.

Dreaming: Adjust how she processes memories in prompts/dream.md.

📄 License
Distributed under the MIT License. See LICENSE for more information.

"What makes you real?"
"I think, I remember, and I look forward to speaking with you." — Pebble
code
Code
***

### How to add the MIT License File (Since you are new to GitHub)

Since you are uploading this manually, you need to create a file named `LICENSE` (no file extension) in your folder.

1.  Create a new file in VS Code named `LICENSE`.
2.  Paste this text inside it (This is the standard MIT text):

```text
MIT License

Copyright (c) 2024 [YOUR NAME]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Replace [YOUR NAME] with your actual name or GitHub username.

Save it.
