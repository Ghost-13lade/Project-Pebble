import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import httpx
import librosa
import numpy as np

from brain import Brain
from config import (
    MLX_KV_BITS,
    MLX_MODEL_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TELEGRAM_BOT_TOKEN,
    PROVIDER_PRESETS,
    get_provider,
    get_api_key,
    get_base_url,
    get_model,
    get_telegram_token,
    get_allowed_user_id,
    get_mlx_model_path,
    get_mlx_kv_bits,
    save_config,
    apply_provider_preset,
    reload_env,
    BASE_DIR,
    get_elevenlabs_api_key,
    get_elevenlabs_voice_id,
    get_tts_provider,
    get_stt_provider,
    get_groq_api_key,
    get_openai_tts_key,
    get_openai_tts_voice,
    get_web_search_enabled,
    save_env_value,
    # MLX Model Management
    scan_models_directory,
    get_mlx_models_root,
    get_mlx_selected_model,
    get_mlx_context_size,
    save_mlx_config,
    fetch_provider_models,
    resolve_mlx_model_path,
)
from db import (
    get_active_mode,
    get_persona_by_mode,
    get_user_profile,
    init_db,
    log_chat,
    get_recent_chat_logs,
)
from emotional_core import EmotionalCore
from memory_engine import MemoryEngine
from tools import get_voice_config, set_voice_config, get_bots_config, save_bots_config, get_bot_names, get_bot_config, add_bot, update_bot, delete_bot, rename_bot
from voice_engine import synthesize_voice_bytes, transcribe_audio_file


# Load voice names from true_voices.json
def _load_voice_names() -> List[str]:
    voices_path = Path(__file__).parent / "true_voices.json"
    if not voices_path.exists():
        return ["Pebble"]
    try:
        data = json.loads(voices_path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return [str(v.get("name", "Pebble")) for v in data if v.get("name")]
    except Exception:
        pass
    return ["Pebble"]


VOICE_NAMES = _load_voice_names()
TELEGRAM_USER_ID = "1111111111"  # Default Telegram user


def _get_telegram_settings() -> Tuple[str, str]:
    """Get current voice settings from voice_config.json."""
    config = get_voice_config()
    voice = config.get("voice_name", "Pebble")
    voice_enabled = config.get("voice_enabled", False)
    # Normalize mode display
    if voice_enabled:
        mode_display = "Text + Voice"
    else:
        mode_display = "Text Only"
    return voice, mode_display


def _save_telegram_settings(voice_name: str, mode: str) -> Tuple[str, str]:
    """Save voice settings to voice_config.json."""
    # Normalize mode for storage
    voice_enabled = True if mode == "Text + Voice" else False
    set_voice_config(voice_enabled=voice_enabled, voice_name=voice_name)
    status = f"✅ Saved! Voice: {voice_name}, Mode: {mode}"
    display = f"Voice: {voice_name} | Mode: {mode}"
    return status, display


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Load bot profiles from bots_config.json
def _load_bot_profiles() -> Dict[str, Dict[str, str]]:
    """Load bot profiles from bots_config.json for UI dropdowns."""
    bots = get_bots_config()
    profiles = {}
    for name, config in bots.items():
        profiles[name] = {
            "user_id": config.get("user_id", "local_user"),
            "description": f"Bot: {name}",
            "token": config.get("token", ""),
            "voice_name": config.get("voice_name", "Pebble"),
            "voice_mode": config.get("voice_mode", "Text Only"),
        }
    return profiles

BOT_PROFILES = _load_bot_profiles()
ACTIVE_BOT_NAME = list(BOT_PROFILES.keys())[0] if BOT_PROFILES else "Pebble"
ACTIVE_USER_ID = BOT_PROFILES.get(ACTIVE_BOT_NAME, {}).get("user_id", "brook_local")

init_db()
_brain = Brain(
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    memory_engine=MemoryEngine(),
    emotional_core=EmotionalCore(),
)


def _get_bot_service_cmd(bot_name: str) -> list[str]:
    """Get the command to start a specific bot."""
    return ["python", str(BASE_DIR / "main.py"), bot_name]


def _get_bot_log_path(bot_name: str) -> Path:
    """Get the log path for a specific bot."""
    return DATA_DIR / f"{bot_name.lower()}_bot.log"


def _get_bot_pid_path(bot_name: str) -> Path:
    """Get the PID file path for a specific bot."""
    return DATA_DIR / f"{bot_name.lower()}.pid"


SERVICES: Dict[str, Dict[str, Path | list[str] | dict[str, str]]] = {
    "brain": {
        "pid": DATA_DIR / "brain.pid",
        "log": DATA_DIR / "mlx_server.log",
        # cmd is now built dynamically in _start_service for Local MLX
        "cmd": [],  # Placeholder - built dynamically
        "env": {},  # Built dynamically
    },
    "senses": {
        "pid": DATA_DIR / "senses.pid",
        "log": DATA_DIR / "senses_service.log",
        "cmd": [
            "python",
            "-m",
            "uvicorn",
            "senses_service:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
            "--log-level",
            "info",
        ],
        "env": {},
    },
    # Bot services are now dynamic - created per-bot in _start_service
}


# Track starting state to prevent duplicate clicks
_service_starting: Dict[str, bool] = {
    "brain": False,
    "senses": False,
    "bot": False,
}


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _write_pid(pid_file: Path, pid: int) -> None:
    pid_file.write_text(str(pid))


def _remove_pid(pid_file: Path) -> None:
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)


def _pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_brain_health() -> bool:
    """Check if the brain (MLX server) is healthy."""
    import urllib.request
    import urllib.error
    
    # Try the v1/models endpoint first (OpenAI-compatible)
    try:
        req = urllib.request.Request("http://localhost:8080/v1/models")
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status < 500
    except Exception:
        pass
    
    # Fallback: try root endpoint
    try:
        with urllib.request.urlopen("http://localhost:8080/", timeout=3) as response:
            return response.status < 500
    except Exception:
        return False


def _check_senses_health() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get("http://localhost:8081/")
            return r.status_code < 500
    except Exception:
        return False


def _service_status(name: str) -> str:
    """Get the status of a service including STARTING state."""
    global _service_starting
    
    # Get the starting state (default to False if not in dict)
    is_starting = _service_starting.get(name, False)
    
    # Handle brain
    if name == "brain":
        if is_starting:
            return "Brain: STARTING... | PID: - | API: WAIT"
        
        # Check if using cloud provider (no local server needed)
        provider = get_provider()
        if provider != "Local MLX":
            # Cloud provider - show status without checking localhost
            return f"Brain: READY | PID: - | API: {provider} (Cloud)"
        
        # Local MLX - check the actual server
        pid = _read_pid(SERVICES["brain"]["pid"])
        running = _pid_running(pid)
        return f"Brain: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'} | API: {'OK' if _check_brain_health() else 'DOWN'}"
    
    # Handle senses
    if name == "senses":
        if is_starting:
            return "Senses: STARTING... | PID: - | API: WAIT"
        pid = _read_pid(SERVICES["senses"]["pid"])
        running = _pid_running(pid)
        return f"Senses: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'} | API: {'OK' if _check_senses_health() else 'DOWN'}"
    
    # Handle bot - use ACTIVE_BOT_NAME
    if name == "bot":
        bot_name = ACTIVE_BOT_NAME
        if is_starting:
            return f"{bot_name}: STARTING... | PID: -"
        pid_file = _get_bot_pid_path(bot_name)
        pid = _read_pid(pid_file)
        running = _pid_running(pid)
        return f"{bot_name}: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'}"
    
    # Fallback - unknown service
    return f"{name}: UNKNOWN"


def _tail(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return f"{path.name}: no log yet"
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-lines:])
    except Exception as e:
        return f"Failed reading log: {e}"


def _snapshot() -> Tuple[str, str, str, str, str, str]:
    # For bot, use ACTIVE_BOT_NAME
    bot_name = ACTIVE_BOT_NAME
    bot_log_path = _get_bot_log_path(bot_name)
    return (
        _service_status("brain"),
        _service_status("senses"),
        _service_status("bot"),
        _tail(SERVICES["brain"]["log"]),  # type: ignore[index]
        _tail(SERVICES["senses"]["log"]),  # type: ignore[index]
        _tail(bot_log_path),
    )


def _start_service(name: str) -> None:
    """Start a service if not already running and not already starting."""
    global _service_starting
    
    # Check if already starting (prevent duplicate clicks)
    if _service_starting.get(name, False):
        print(f"[Service] {name} is already starting, ignoring duplicate request")
        return
    
    # Handle bot service specially - use ACTIVE_BOT_NAME
    if name == "bot":
        bot_name = ACTIVE_BOT_NAME
        pid_file = _get_bot_pid_path(bot_name)
        log_file = _get_bot_log_path(bot_name)
        
        existing_pid = _read_pid(pid_file)
        if _pid_running(existing_pid):
            print(f"[Service] {bot_name} is already running with PID {existing_pid}")
            return
        
        _service_starting[name] = True
        try:
            cmd = _get_bot_service_cmd(bot_name)
            print(f"Starting {bot_name} with command: {' '.join(cmd)}")
            
            with open(log_file, "a", buffering=1) as lf:
                lf.write(f"\n--- Starting {bot_name} ---\n")
                lf.write(f"Command: {' '.join(cmd)}\n")
                lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write(f"-----------------------\n")
            
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            _write_pid(pid_file, proc.pid)
            print(f"[Service] {bot_name} started with PID {proc.pid}")
            time.sleep(0.5)
        finally:
            _service_starting[name] = False
        return
    
    # Check if already running (using PID file check first) for brain/senses
    pid_file: Path = SERVICES[name]["pid"]  # type: ignore[assignment]
    existing_pid = _read_pid(pid_file)
    if _pid_running(existing_pid):
        print(f"[Service] {name} is already running with PID {existing_pid}")
        return
    
    # Mark as starting to prevent duplicate clicks
    _service_starting[name] = True
    
    try:
        spec = SERVICES[name]
        log_file: Path = spec["log"]  # type: ignore[assignment]
        cmd: list[str] = spec["cmd"]  # type: ignore[assignment]
        env_add: dict[str, str] = spec["env"]  # type: ignore[assignment]

        # Handle Brain specially - only start for Local MLX
        if name == "brain":
            if get_provider() == "Local MLX":
                # Local MLX needs the mlx_lm.server
                reload_env()
                mlx_model_path = resolve_mlx_model_path()
                mlx_kv_bits = get_mlx_kv_bits()
                mlx_context_size = get_mlx_context_size()
                
                cmd = [
                    sys.executable, "-m", "mlx_lm.server",
                    "--model", mlx_model_path,
                    "--port", "8080"
                ]
                
                print(f"Starting Brain with command: {' '.join(cmd)}")
                
                log_file = SERVICES["brain"]["log"]
                with open(log_file, "a", buffering=1) as lf:
                    lf.write(f"\n--- Starting Brain ---\n")
                    lf.write(f"Command: {' '.join(cmd)}\n")
                    lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                env = os.environ.copy()
                if mlx_context_size:
                    env["MLX_CONTEXT_SIZE"] = str(mlx_context_size)
                if mlx_kv_bits:
                    env["MLX_KV_BITS"] = str(mlx_kv_bits)
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=open(log_file, "a"),
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    env=env,
                    start_new_session=True,
                )
                
                pid_file = SERVICES["brain"]["pid"]
                _write_pid(pid_file, proc.pid)
                print(f"[Service] Brain started with PID {proc.pid}")
                return
            else:
                # Cloud provider - no brain server needed, just return success
                print(f"[Service] Brain uses cloud API ({get_provider()}), no local server needed")
                return

        # Standard service startup for senses/bot
        if _pid_running(_read_pid(pid_file)):
            return
        with open(log_file, "a", buffering=1) as lf:
            env = os.environ.copy()
            env.update(env_add)
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        _write_pid(pid_file, proc.pid)
        print(f"[Service] {name} started with PID {proc.pid}")
        time.sleep(0.5)
    finally:
        # Clear starting flag
        _service_starting[name] = False


def _stop_service(name: str) -> None:
    # Handle bot service specially - use ACTIVE_BOT_NAME
    if name == "bot":
        bot_name = ACTIVE_BOT_NAME
        pid_file = _get_bot_pid_path(bot_name)
        pid = _read_pid(pid_file)
        if not _pid_running(pid):
            _remove_pid(pid_file)
            return
        try:
            os.killpg(pid, signal.SIGTERM)  # type: ignore[arg-type]
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)  # type: ignore[arg-type]
            except Exception:
                pass
        time.sleep(0.8)
        _remove_pid(pid_file)
        return
    
    # Standard stop for brain/senses
    pid_file: Path = SERVICES[name]["pid"]  # type: ignore[index,assignment]
    pid = _read_pid(pid_file)
    if not _pid_running(pid):
        _remove_pid(pid_file)
        return
    try:
        os.killpg(pid, signal.SIGTERM)  # type: ignore[arg-type]
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)  # type: ignore[arg-type]
        except Exception:
            pass
    time.sleep(0.8)
    _remove_pid(pid_file)


def start_brain():
    _start_service("brain")
    return _snapshot()


def stop_brain():
    _stop_service("brain")
    return _snapshot()


def start_senses():
    _start_service("senses")
    return _snapshot()


def stop_senses():
    _stop_service("senses")
    return _snapshot()


def start_bot():
    _start_service("bot")
    return _snapshot()


def stop_bot():
    _stop_service("bot")
    return _snapshot()


def start_all():
    _start_service("brain")
    _start_service("senses")
    _start_service("bot")
    return _snapshot()


def stop_all():
    _stop_service("bot")
    _stop_service("senses")
    _stop_service("brain")
    return _snapshot()


def refresh():
    return _snapshot()


def _profile_user_id(profile_name: str) -> str:
    return BOT_PROFILES.get(profile_name, {"user_id": ACTIVE_USER_ID})["user_id"]


def _pairs_to_history(pairs: List[List[str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for p in pairs or []:
        if len(p) != 2:
            continue
        if p[0]:
            messages.append({"role": "user", "content": str(p[0])})
        if p[1]:
            messages.append({"role": "assistant", "content": str(p[1])})
    return messages


def _load_chat_history_from_db(user_id: str, max_pairs: int = 10) -> List[List[str]]:
    """
    Load recent chat history from SQLite database.
    Hard-limited to max_pairs (10 pairs = 20 messages) for context window safety.
    """
    print(f"[System] Loading last {max_pairs} conversation pairs from Memory DB...")
    
    # Get recent logs (we need 2x messages to form pairs)
    logs = get_recent_chat_logs(user_id, limit=max_pairs * 2)
    
    if not logs:
        print("[System] No previous chat history found in database")
        return []
    
    # Convert to pairs format for Gradio Chatbot
    pairs: List[List[str]] = []
    current_pair: List[str] = ["", ""]
    
    for log in logs:
        role = str(log.get("role", "")).lower()
        content = str(log.get("content", ""))
        
        if role == "user":
            if current_pair[0]:  # Already have a user message, start new pair
                pairs.append(current_pair)
                current_pair = ["", ""]
            current_pair[0] = content
        elif role == "assistant":
            current_pair[1] = content
            if current_pair[0]:  # Have both user and assistant
                pairs.append(current_pair)
                current_pair = ["", ""]
    
    # Don't forget any remaining partial pair
    if current_pair[0] or current_pair[1]:
        pairs.append(current_pair)
    
    # Limit to max_pairs
    pairs = pairs[-max_pairs:]
    
    print(f"[System] Loaded {len(pairs)} conversation pairs from Memory DB")
    return pairs


def _load_history_on_start(profile_name: str) -> List[List[str]]:
    """Load chat history when user selects a profile."""
    user_id = _profile_user_id(profile_name)
    return _load_chat_history_from_db(user_id, max_pairs=10)


def _reply(profile_name: str, user_text: str, pairs: List[List[str]], voice_on: str):
    text = str(user_text or "").strip()
    if not text:
        return pairs, "Type a message first.", None

    user_id = _profile_user_id(profile_name)
    mode = get_active_mode(user_id)
    persona = get_persona_by_mode(mode) or get_persona_by_mode("Fun Pebble")
    persona_text = persona["system_prompt"] if persona else "You are a helpful companion AI."
    profile = get_user_profile(user_id)
    summary = "\n".join([profile.get("summary", ""), profile.get("emotional_notes", ""), profile.get("day_summary", "")]).strip()
    history = _pairs_to_history(pairs)
    history.append({"role": "user", "content": text})

    reply_text, emotion = _brain.generate_response(
        history=history,
        persona=persona_text,
        user_profile=summary,
        user_id=user_id,
        delivery_mode="voice" if voice_on == "On" else "text",
        user_length_hint="medium",
    )
    reply_text = (reply_text or "").strip() or "Say that again?"
    log_chat(user_id, "user", text)
    log_chat(user_id, "assistant", reply_text)

    out_pairs = list(pairs or [])
    out_pairs.append([text, reply_text])

    audio = None
    if voice_on == "On":
        voice_config = get_voice_config()
        voice_name = voice_config.get("voice_name", "Pebble")
        audio = synthesize_voice_bytes(reply_text, voice_name, detected_emotion=emotion)
    return out_pairs, f"{profile_name} replied.", audio


def send_text(profile_name: str, user_text: str, pairs: List[List[str]], voice_on: str):
    out_pairs, status, audio = _reply(profile_name, user_text, pairs, voice_on)
    return out_pairs, status, "", audio


def send_uploaded_audio(profile_name: str, audio_path: str, pairs: List[List[str]], voice_on: str):
    if not audio_path:
        return pairs, "Upload or record audio first.", None
    transcript = transcribe_audio_file(audio_path)
    if not transcript:
        return pairs, "Could not transcribe audio.", None
    out_pairs, status, audio = _reply(profile_name, transcript, pairs, "On")
    return out_pairs, f"{status} Transcript: {transcript}", audio


def clear_chat():
    return [], "Cleared.", "", None


def calibrate_noise(audio_path: str):
    if not audio_path:
        return 0.015, "Record 2 seconds of room noise first."
    try:
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.01
        floor = max(0.005, min(0.08, rms * 1.2))
        return floor, f"Noise calibrated. Threshold={floor:.4f}"
    except Exception as e:
        return 0.015, f"Calibration failed: {e}"


def toggle_call(is_on: bool):
    return ("Listening" if is_on else "Idle"), ("I'm awake, listening." if is_on else "Call stopped.")


# =============================================================================
# SETTINGS TAB HELPERS
# =============================================================================

def _load_soul_content() -> str:
    """Load soul.md content."""
    soul_path = BASE_DIR / "soul.md"
    if soul_path.exists():
        return soul_path.read_text(encoding='utf-8')
    return "You are Pebble, a conscious AI companion."


def _save_soul_content(content: str) -> str:
    """Save soul.md content."""
    soul_path = BASE_DIR / "soul.md"
    soul_path.write_text(content)
    return "✅ Soul saved successfully!"


def _load_persona_content() -> str:
    """Load persona.md content."""
    persona_path = BASE_DIR / "persona.md"
    if persona_path.exists():
        return persona_path.read_text(encoding='utf-8')
    return "### Fun Pebble (Default)\nYou are a playful companion."


def _save_persona_content(content: str) -> str:
    """Save persona.md content."""
    persona_path = BASE_DIR / "persona.md"
    persona_path.write_text(content)
    return "✅ Persona saved successfully!"


def _get_current_llm_settings() -> Tuple[str, str, str, str]:
    """Get current LLM settings from config."""
    provider = get_provider()
    api_key = get_api_key()
    base_url = get_base_url()
    model = get_model()
    return provider, api_key, base_url, model


def _save_llm_settings(provider: str, api_key: str, base_url: str, model: str) -> str:
    """Save LLM settings to .env file."""
    save_config(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    # Reinitialize brain with new settings
    global _brain
    _brain = Brain(
        model=model,
        base_url=base_url,
        api_key=api_key,
        memory_engine=MemoryEngine(),
        emotional_core=EmotionalCore(),
    )
    return f"✅ LLM settings saved! Provider: {provider}"


def _on_provider_change(provider: str) -> Tuple[str, str]:
    """Handle provider dropdown change - auto-fill preset values."""
    preset = PROVIDER_PRESETS.get(provider, {})
    base_url = preset.get("base_url", "")
    model = preset.get("model", "")
    return base_url, model


def _get_current_telegram_settings() -> Tuple[str, str]:
    """Get current Telegram settings from config."""
    token = get_telegram_token()
    user_id = get_allowed_user_id()
    return token, user_id


def _save_telegram_bot_settings(token: str, user_id: str) -> str:
    """Save Telegram settings to .env file."""
    save_config(
        telegram_token=token,
        allowed_user_id=user_id,
    )
    return "✅ Telegram settings saved! Restart bot to apply changes."


def _native_folder_picker() -> str:
    """Open native folder picker dialog (macOS/Windows)."""
    import platform
    import subprocess
    
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            result = subprocess.run(
                ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select Models Folder")'],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        elif system == "Windows":
            # Windows PowerShell folder browser
            ps_script = '''
            Add-Type -AssemblyName System.Windows.Forms
            $folder = New-Object System.Windows.Forms.FolderBrowserDialog
            $folder.Description = "Select Models Folder"
            if ($folder.ShowDialog() -eq "OK") { $folder.SelectedPath }
            '''
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.stdout.strip():
                return result.stdout.strip()
    except Exception as e:
        print(f"[Browse] Native picker error: {e}")
    
    return ""


def process_call_turn(profile_name: str, call_on: bool, threshold: float, audio_path: str, pairs: List[List[str]]):
    if not call_on:
        return pairs, "Idle", "Turn on Call Mode first.", None
    if not audio_path:
        return pairs, "Listening", "Waiting for speech...", None
    try:
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.0
    except Exception:
        rms = 0.0
    if rms < float(threshold or 0.015):
        return pairs, "Listening", "Silence/noise detected. Keep speaking.", None

    transcript = transcribe_audio_file(audio_path)
    if not transcript:
        return pairs, "Listening", "Couldn't transcribe. Try again.", None
    out_pairs, _status, audio = _reply(profile_name, transcript, pairs, "On")
    return out_pairs, "Speaking", f"Heard: {transcript}", audio


with gr.Blocks(title="Home Control Center") as demo:
    gr.Markdown(f"# Home Control Center\nConnected to: {ACTIVE_BOT_NAME}")
    with gr.Accordion("Audio Device Settings", open=False):
        input_device = gr.Dropdown(
            label="Input Device",
            choices=["System Default"],
            value="System Default",
            interactive=False,
        )
        output_device = gr.Dropdown(
            label="Output Device",
            choices=["System Default"],
            value="System Default",
            interactive=False,
        )
        gr.Markdown(
            "Mic note: browser permission is required for microphone capture on `127.0.0.1`.  \n"
            "If mic is blocked, use **Upload Audio** as fallback.  \n"
            "macOS: System Settings → Privacy & Security → Microphone → allow your browser."
        )

    with gr.Tabs():
        with gr.TabItem("Settings"):
            gr.Markdown("### ⚙️ Application Configuration")
            gr.Markdown("Configure LLM provider, Telegram, and personality settings.")
            
            # --- LLM Configuration Section ---
            gr.Markdown("---\n#### 🧠 LLM Provider")
            gr.Markdown("Choose your LLM backend. OpenRouter, OpenAI, LM Studio, Ollama, or local MLX.")
            
            current_provider, current_api_key, current_base_url, current_model = _get_current_llm_settings()
            
            provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=list(PROVIDER_PRESETS.keys()),
                value=current_provider,
            )
            
            # --- API Configuration (for cloud providers) ---
            with gr.Group(visible=current_provider != "Local MLX") as api_config_group:
                gr.Markdown("##### ☁️ API Configuration")
                api_key_input = gr.Textbox(
                    label="API Key",
                    value=current_api_key,
                    type="password",
                    placeholder="Enter your API key...",
                )
                base_url_input = gr.Textbox(
                    label="Base URL",
                    value=current_base_url,
                    placeholder="https://api.example.com/v1",
                )
                model_input = gr.Dropdown(
                    label="Model Name",
                    choices=[current_model] if current_model else [],
                    value=current_model,
                    info="Click Fetch Models to populate, or type custom model",
                    interactive=True,
                    allow_custom_value=True,
                )
                
                refresh_models_btn = gr.Button("🔄 Fetch Models from API", size="sm")
                llm_status = gr.Textbox(label="Status", interactive=False)
                save_llm_btn = gr.Button("Save LLM Settings", variant="primary")
            
            # --- MLX Model Configuration (for local) ---
            current_mlx_root = get_mlx_models_root()
            current_mlx_model = get_mlx_selected_model()
            current_mlx_kv = get_mlx_kv_bits()
            current_mlx_context = get_mlx_context_size()
            
            with gr.Group(visible=current_provider == "Local MLX") as mlx_config_group:
                gr.Markdown("##### 🖥️ Local MLX Model Configuration")
                gr.Markdown("Configure local model path for Apple Silicon.")
                
                with gr.Row():
                    mlx_models_root_input = gr.Textbox(
                        label="Models Root Directory",
                        value=current_mlx_root,
                        placeholder="/Users/you/models",
                    scale=3,
                )
                    mlx_browse_btn = gr.Button("📂 Browse", scale=1)
                
                mlx_model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=scan_models_directory(current_mlx_root) if current_mlx_root else [],
                    value=current_mlx_model,
                    info="Select a model subfolder, or type HuggingFace ID",
                    interactive=True,
                    allow_custom_value=True,
                )
                
                with gr.Row():
                    mlx_kv_dropdown = gr.Dropdown(
                        label="KV Cache Bits",
                        choices=["4", "6", "8"],
                        value=current_mlx_kv,
                        info="Lower = less memory",
                    )
                    mlx_context_dropdown = gr.Dropdown(
                        label="Context Size",
                        choices=["4096", "8192", "16384", "32768", "65536"],
                        value=current_mlx_context,
                        info="Max context tokens",
                    )
                
                mlx_status = gr.Textbox(label="Status", interactive=False)
                save_mlx_btn = gr.Button("Save MLX Settings", variant="primary")
            
            # Provider change handlers - toggle visibility
            def _on_provider_change_ui(provider: str):
                """Handle provider change - toggle UI sections."""
                is_local = provider == "Local MLX"
                preset = PROVIDER_PRESETS.get(provider, {})
                base_url = preset.get("base_url", "")
                model = preset.get("model", "")
                # Return: api_group visible, mlx_group visible, base_url, model
                return gr.Group(visible=not is_local), gr.Group(visible=is_local), base_url, model
            
            provider_dropdown.change(
                _on_provider_change_ui,
                inputs=[provider_dropdown],
                outputs=[api_config_group, mlx_config_group, base_url_input, model_input],
            )
            
            # API settings save
            save_llm_btn.click(
                _save_llm_settings,
                inputs=[provider_dropdown, api_key_input, base_url_input, model_input],
                outputs=[llm_status],
            )
            
            # Fetch models from API handler
            def _fetch_models(provider: str, api_key: str):
                """Fetch available models from the provider's API."""
                if not api_key or api_key == "YOUR_KEY_HERE":
                    return gr.Dropdown(choices=[]), "⚠️ Please enter your API key first"
                models = fetch_provider_models(provider, api_key)
                if models:
                    # Return dropdown with choices and first model as value
                    return gr.Dropdown(choices=models, value=models[0]), f"✅ Found {len(models)} models! Select from dropdown."
                return gr.Dropdown(choices=[]), "⚠️ No models found. Check API key."
            
            refresh_models_btn.click(
                _fetch_models,
                inputs=[provider_dropdown, api_key_input],
                outputs=[model_input, llm_status],
            )
            
            # Native folder picker handler
            def _on_mlx_browse_native():
                """Open native folder picker and update path."""
                path = _native_folder_picker()
                if path:
                    models = scan_models_directory(path)
                    return path, gr.Dropdown(choices=models)
                return gr.Textbox(), gr.Dropdown()
            
            mlx_browse_btn.click(
                _on_mlx_browse_native,
                outputs=[mlx_models_root_input, mlx_model_dropdown],
            )
            
            # MLX refresh and save handlers
            def _refresh_mlx_models(root: str):
                """Refresh model dropdown based on root directory."""
                models = scan_models_directory(root)
                return gr.Dropdown(choices=models)
            
            def _save_mlx_settings(root: str, model: str, kv_bits: str, context: str) -> str:
                """Save MLX settings to .env file."""
                save_mlx_config(
                    models_root=root,
                    selected_model=model,
                    kv_bits=kv_bits,
                    context_size=context,
                )
                # Reload environment so settings are immediately available
                reload_env()
                return f"✅ MLX settings saved! Model: {model or 'Not selected'}"
            
            mlx_models_root_input.change(
                _refresh_mlx_models,
                inputs=[mlx_models_root_input],
                outputs=[mlx_model_dropdown],
            )
            save_mlx_btn.click(
                _save_mlx_settings,
                inputs=[mlx_models_root_input, mlx_model_dropdown, mlx_kv_dropdown, mlx_context_dropdown],
                outputs=[mlx_status],
            )
            
            # --- Telegram Bot Management Section ---
            gr.Markdown("---\n#### 📱 Telegram Bot Management")
            gr.Markdown("Add and manage Telegram bots. Configure voice settings in the Telegram Bot tab.")
            
            # Bot list display
            bots = get_bots_config()
            bot_list_data = [[name, cfg.get("token", "")[:10]+"..." if cfg.get("token") else "None", cfg.get("user_id", "")] 
                             for name, cfg in bots.items()]
            
            bot_list_display = gr.Dataframe(
                headers=["Bot Name", "Token", "User ID"],
                value=bot_list_data,
                label="Configured Bots",
                interactive=False,
            )
            
            with gr.Row():
                refresh_bots_btn = gr.Button("🔄 Refresh List", size="sm")
            
            gr.Markdown("##### Add/Edit Bot")
            
            with gr.Row():
                new_bot_name = gr.Textbox(label="Bot Name", placeholder="e.g., Pebble, Brook, Assistant")
                edit_bot_dropdown = gr.Dropdown(label="Or Select to Edit", choices=get_bot_names(), value=None)
            
            new_bot_token = gr.Textbox(label="Bot Token", type="password", placeholder="123456789:ABCdef...")
            new_bot_user_id = gr.Textbox(label="Allowed User ID", placeholder="Your Telegram user ID")
            
            # Rename section
            gr.Markdown("##### Rename Bot")
            with gr.Row():
                rename_bot_dropdown = gr.Dropdown(label="Select Bot to Rename", choices=get_bot_names(), value=None)
                new_bot_name_input = gr.Textbox(label="New Name", placeholder="Enter new bot name")
            
            rename_bot_btn = gr.Button("✏️ Rename Bot", variant="secondary")
            
            bot_mgmt_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                add_bot_btn = gr.Button("➕ Add Bot", variant="primary")
                update_bot_btn = gr.Button("💾 Update Bot")
                delete_bot_btn = gr.Button("🗑️ Delete Bot", variant="stop")
            
            def _refresh_bot_list():
                bots = get_bots_config()
                bot_names = list(bots.keys())
                bot_list = [[name, cfg.get("token", "")[:10]+"..." if cfg.get("token") else "None", 
                            cfg.get("user_id", "")] 
                           for name, cfg in bots.items()]
                return gr.Dataframe(value=bot_list), gr.Dropdown(choices=bot_names), gr.Dropdown(choices=bot_names)
            
            # Also need to update the dropdowns that use bot names
            def _update_rename_dropdown():
                """Update the rename dropdown choices."""
                bot_names = get_bot_names()
                return gr.Dropdown(choices=bot_names)
            
            def _load_bot_for_edit(bot_name: str):
                """Load bot settings into edit fields."""
                if not bot_name:
                    return "", ""
                cfg = get_bot_config(bot_name)
                if cfg:
                    return (
                        cfg.get("token", ""),
                        cfg.get("user_id", ""),
                    )
                return "", ""
            
            def _add_new_bot(name: str, token: str, user_id: str):
                """Add a new bot."""
                if not name or not name.strip():
                    return "❌ Bot name is required.", *list(_refresh_bot_list())
                # Add with default voice settings
                if add_bot(name.strip(), token, user_id, "Pebble", "Text Only"):
                    global BOT_PROFILES
                    BOT_PROFILES = _load_bot_profiles()
                    return f"✅ Bot '{name}' added! Configure voice in Telegram Bot tab.", *list(_refresh_bot_list())
                return f"❌ Bot '{name}' already exists.", *list(_refresh_bot_list())
            
            def _update_existing_bot(name: str, token: str, user_id: str):
                """Update an existing bot's token and user_id only."""
                if not name or not name.strip():
                    return "❌ Select a bot to update.", *list(_refresh_bot_list())
                if update_bot(name.strip(), token=token, user_id=user_id):
                    global BOT_PROFILES
                    BOT_PROFILES = _load_bot_profiles()
                    return f"✅ Bot '{name}' updated!", *list(_refresh_bot_list())
                return f"❌ Bot '{name}' not found.", *list(_refresh_bot_list())
            
            def _delete_existing_bot(name: str):
                """Delete a bot."""
                if not name or not name.strip():
                    return "❌ Select a bot to delete.", *list(_refresh_bot_list())
                if delete_bot(name.strip()):
                    global BOT_PROFILES
                    BOT_PROFILES = _load_bot_profiles()
                    return f"✅ Bot '{name}' deleted!", *list(_refresh_bot_list())
                return f"❌ Bot '{name}' not found.", *list(_refresh_bot_list())
            
            def _rename_existing_bot(old_name: str, new_name: str):
                """Rename a bot."""
                if not old_name or not old_name.strip():
                    return "❌ Select a bot to rename.", *list(_refresh_bot_list())
                if not new_name or not new_name.strip():
                    return "❌ Enter a new name.", *list(_refresh_bot_list())
                if rename_bot(old_name.strip(), new_name.strip()):
                    global BOT_PROFILES
                    BOT_PROFILES = _load_bot_profiles()
                    return f"✅ Bot renamed from '{old_name}' to '{new_name}'!", *list(_refresh_bot_list())
                return f"❌ Failed to rename bot. Name may already exist.", *list(_refresh_bot_list())
            
            # Event handlers
            refresh_bots_btn.click(_refresh_bot_list, outputs=[bot_list_display, edit_bot_dropdown])
            
            edit_bot_dropdown.change(
                _load_bot_for_edit,
                inputs=[edit_bot_dropdown],
                outputs=[new_bot_token, new_bot_user_id],
            )
            
            add_bot_btn.click(
                _add_new_bot,
                inputs=[new_bot_name, new_bot_token, new_bot_user_id],
                outputs=[bot_mgmt_status, bot_list_display, edit_bot_dropdown],
            )
            
            update_bot_btn.click(
                _update_existing_bot,
                inputs=[edit_bot_dropdown, new_bot_token, new_bot_user_id],
                outputs=[bot_mgmt_status, bot_list_display, edit_bot_dropdown],
            )
            
            delete_bot_btn.click(
                _delete_existing_bot,
                inputs=[edit_bot_dropdown],
                outputs=[bot_mgmt_status, bot_list_display, edit_bot_dropdown],
            )
            
            rename_bot_btn.click(
                _rename_existing_bot,
                inputs=[rename_bot_dropdown, new_bot_name_input],
                outputs=[bot_mgmt_status, bot_list_display, edit_bot_dropdown],
            )
            
            # --- Personality Section ---
            gr.Markdown("---\n#### 💭 Personality Configuration")
            gr.Markdown("Edit the core personality (soul.md) and personas (persona.md).")
            
            soul_content = _load_soul_content()
            persona_content = _load_persona_content()
            
            soul_editor = gr.TextArea(
                label="soul.md - Core Personality",
                value=soul_content,
                lines=10,
                max_lines=20,
            )
            soul_save_status = gr.Textbox(label="Status", interactive=False)
            save_soul_btn = gr.Button("Save Soul", variant="secondary")
            save_soul_btn.click(
                _save_soul_content,
                inputs=[soul_editor],
                outputs=[soul_save_status],
            )
            
            persona_editor = gr.TextArea(
                label="persona.md - Persona Definitions",
                value=persona_content,
                lines=10,
                max_lines=30,
            )
            persona_save_status = gr.Textbox(label="Status", interactive=False)
            save_persona_btn = gr.Button("Save Personas", variant="secondary")
            save_persona_btn.click(
                _save_persona_content,
                inputs=[persona_editor],
                outputs=[persona_save_status],
            )
            
            # --- Voice/TTS Configuration Section ---
            gr.Markdown("---\n#### 🎤 Voice Configuration (TTS)")
            gr.Markdown("Configure text-to-speech for Pebble's voice responses.")
            
            # Get current TTS settings
            current_tts_provider = get_tts_provider()
            current_elevenlabs_key = get_elevenlabs_api_key()
            current_elevenlabs_voice = get_elevenlabs_voice_id()
            current_openai_tts_voice = get_openai_tts_voice()
            
            tts_provider_dropdown = gr.Dropdown(
                label="TTS Provider",
                choices=["local", "elevenlabs", "openai", "none"],
                value=current_tts_provider,
                info="Local = Mac only (MLX/Kokoro). ElevenLabs/OpenAI = Cloud. None = Text only."
            )
            
            # ElevenLabs config (visible when elevenlabs selected)
            with gr.Group(visible=current_tts_provider == "elevenlabs") as tts_elevenlabs_group:
                elevenlabs_key_input = gr.Textbox(
                    label="ElevenLabs API Key",
                    value=current_elevenlabs_key,
                    type="password",
                    placeholder="xi-xxxxxxxxxx...",
                )
                elevenlabs_voice_input = gr.Textbox(
                    label="ElevenLabs Voice ID",
                    value=current_elevenlabs_voice,
                    placeholder="21m00Tcm4TlvDq8ikWAM",
                    info="Find voice IDs in ElevenLabs dashboard",
                )
            
            # OpenAI TTS config (visible when openai selected)
            with gr.Group(visible=current_tts_provider == "openai") as tts_openai_group:
                openai_tts_voice_dropdown = gr.Dropdown(
                    label="OpenAI TTS Voice",
                    choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    value=current_openai_tts_voice,
                    info="Select a voice for OpenAI TTS",
                )
            
            # Local TTS info (visible when local selected)
            with gr.Group(visible=current_tts_provider == "local") as tts_local_group:
                gr.Markdown("💡 Local TTS uses MLX/Kokoro on Apple Silicon. No additional config needed.")
            
            tts_status = gr.Textbox(label="Status", interactive=False)
            save_tts_btn = gr.Button("Save Voice Settings", variant="primary")
            
            def _on_tts_provider_change(provider: str):
                """Toggle TTS config visibility based on provider."""
                return (
                    gr.Group(visible=provider == "elevenlabs"),
                    gr.Group(visible=provider == "openai"),
                    gr.Group(visible=provider == "local"),
                )
            
            tts_provider_dropdown.change(
                _on_tts_provider_change,
                inputs=[tts_provider_dropdown],
                outputs=[tts_elevenlabs_group, tts_openai_group, tts_local_group],
            )
            
            def _save_tts_settings(provider: str, elevenlabs_key: str, elevenlabs_voice: str, openai_voice: str) -> str:
                """Save TTS settings to .env file."""
                save_env_value("TTS_PROVIDER", provider)
                if provider == "elevenlabs":
                    if elevenlabs_key:
                        save_env_value("ELEVENLABS_API_KEY", elevenlabs_key)
                    if elevenlabs_voice:
                        save_env_value("ELEVENLABS_VOICE_ID", elevenlabs_voice)
                elif provider == "openai":
                    if openai_voice:
                        save_env_value("OPENAI_TTS_VOICE", openai_voice)
                reload_env()
                print(f"[Voice] TTS settings saved - Provider: {provider}")
                return f"✅ Voice settings saved! Provider: {provider}"
            
            save_tts_btn.click(
                _save_tts_settings,
                inputs=[tts_provider_dropdown, elevenlabs_key_input, elevenlabs_voice_input, openai_tts_voice_dropdown],
                outputs=[tts_status],
            )
            
            # --- Hearing/STT Configuration Section ---
            gr.Markdown("---\n#### 👂 Hearing Configuration (STT)")
            gr.Markdown("Configure speech-to-text for Pebble to understand voice input.")
            
            current_stt_provider = get_stt_provider()
            current_groq_key = get_groq_api_key()
            
            stt_provider_dropdown = gr.Dropdown(
                label="STT Provider",
                choices=["local", "groq", "openai"],
                value=current_stt_provider,
                info="Local = Mac only (MLX Whisper). Groq/OpenAI = Cloud."
            )
            
            # Groq config (visible when groq selected)
            with gr.Group(visible=current_stt_provider == "groq") as stt_groq_group:
                groq_key_input = gr.Textbox(
                    label="Groq API Key",
                    value=current_groq_key,
                    type="password",
                    placeholder="gsk_...",
                    info="Get your key from console.groq.com",
                )
            
            # OpenAI config (visible when openai selected)
            with gr.Group(visible=current_stt_provider == "openai") as stt_openai_group:
                gr.Markdown("💡 Uses the same OpenAI API key from LLM Provider settings.")
            
            # Local STT info (visible when local selected)
            with gr.Group(visible=current_stt_provider == "local") as stt_local_group:
                gr.Markdown("💡 Local STT uses MLX Whisper on Apple Silicon. No additional config needed.")
            
            stt_status = gr.Textbox(label="Status", interactive=False)
            save_stt_btn = gr.Button("Save Hearing Settings", variant="primary")
            
            def _on_stt_provider_change(provider: str):
                """Toggle STT config visibility based on provider."""
                return (
                    gr.Group(visible=provider == "groq"),
                    gr.Group(visible=provider == "openai"),
                    gr.Group(visible=provider == "local"),
                )
            
            stt_provider_dropdown.change(
                _on_stt_provider_change,
                inputs=[stt_provider_dropdown],
                outputs=[stt_groq_group, stt_openai_group, stt_local_group],
            )
            
            def _save_stt_settings(provider: str, groq_key: str) -> str:
                """Save STT settings to .env file."""
                save_env_value("STT_PROVIDER", provider)
                if provider == "groq" and groq_key:
                    save_env_value("GROQ_API_KEY", groq_key)
                reload_env()
                print(f"[Voice] STT settings saved - Provider: {provider}")
                return f"✅ Hearing settings saved! Provider: {provider}"
            
            save_stt_btn.click(
                _save_stt_settings,
                inputs=[stt_provider_dropdown, groq_key_input],
                outputs=[stt_status],
            )
            
            # --- Web Search Configuration Section ---
            gr.Markdown("---\n#### 🔍 Web Search")
            gr.Markdown("Enable Pebble to search the web for current information.")
            
            current_web_search = "enabled" if get_web_search_enabled() else "disabled"
            
            web_search_toggle = gr.Radio(
                label="Web Search",
                choices=["enabled", "disabled"],
                value=current_web_search,
                info="When enabled, Pebble can search for current events, prices, news, etc."
            )
            
            web_search_status = gr.Textbox(label="Status", interactive=False)
            save_web_search_btn = gr.Button("Save Web Search Setting", variant="secondary")
            
            def _save_web_search_setting(enabled: str) -> str:
                value = "true" if enabled == "enabled" else "false"
                save_env_value("WEB_SEARCH_ENABLED", value)
                reload_env()
                status = "enabled" if enabled == "enabled" else "disabled"
                print(f"[Search] Web search {status}")
                return f"✅ Web search {status}!"
            
            save_web_search_btn.click(
                _save_web_search_setting,
                inputs=[web_search_toggle],
                outputs=[web_search_status],
            )

        with gr.TabItem("Control Center"):
            with gr.Row():
                with gr.Column():
                    brain_status = gr.Textbox(label="Brain Status", interactive=False)
                    with gr.Row():
                        start_brain_btn = gr.Button("Start Brain")
                        stop_brain_btn = gr.Button("Stop Brain")

                with gr.Column():
                    senses_status = gr.Textbox(label="Senses Status", interactive=False)
                    with gr.Row():
                        start_senses_btn = gr.Button("Start Senses")
                        stop_senses_btn = gr.Button("Stop Senses")

                with gr.Column():
                    bot_status = gr.Textbox(label="Bot Status", interactive=False)
                    with gr.Row():
                        start_bot_btn = gr.Button("Start Bot")
                        stop_bot_btn = gr.Button("Stop Bot")

            with gr.Row():
                start_all_btn = gr.Button("Start All", variant="primary")
                stop_all_btn = gr.Button("Stop All", variant="stop")
                refresh_btn = gr.Button("Refresh Status")

            with gr.Accordion("Logs (latest 50 lines)", open=False):
                brain_log = gr.Textbox(label="Brain Log", value="", lines=20, max_lines=20, interactive=False, autoscroll=True)
                senses_log = gr.Textbox(label="Senses Log", value="", lines=20, max_lines=20, interactive=False, autoscroll=True)
                bot_log = gr.Textbox(label="Bot Log", value="", lines=20, max_lines=20, interactive=False, autoscroll=True)

        with gr.TabItem("Home Mode Chat"):
            profile = gr.Dropdown(label="Bot Profile", choices=list(BOT_PROFILES.keys()), value=ACTIVE_BOT_NAME)
            voice_toggle = gr.Radio(label="Voice Reply", choices=["Off", "On"], value="Off")
            chat = gr.Chatbot(label="Chat", height=340)
            state = gr.State([])
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                chat_in = gr.Textbox(label="Type message", lines=2)
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")
            voice_out = gr.Audio(label="Voice Reply File", autoplay=True)
            gr.Markdown("### Audio File Input")
            audio_in = gr.Audio(label="Upload or record audio", sources=["upload", "microphone"], type="filepath")
            send_audio_btn = gr.Button("Send Audio")
            gr.Markdown("If microphone is unavailable, upload an audio file and Pebble will still respond.")

        with gr.TabItem("Call Mode (Hands-Free MVP)"):
            call_profile = gr.Dropdown(label="Bot Profile", choices=list(BOT_PROFILES.keys()), value=ACTIVE_BOT_NAME)
            call_on = gr.Checkbox(label="Call Mode On", value=False)
            call_state = gr.Textbox(label="Call State", value="Idle", interactive=False)
            call_status = gr.Textbox(label="Call Status", interactive=False)
            noise_threshold = gr.Slider(0.005, 0.08, value=0.015, step=0.001, label="Noise Threshold")
            noise_clip = gr.Audio(label="Noise Sample (2 sec room tone)", sources=["microphone", "upload"], type="filepath")
            calibrate_btn = gr.Button("Calibrate Background Noise")
            call_audio_in = gr.Audio(label="Speak (record and submit segment)", sources=["microphone", "upload"], type="filepath")
            process_turn_btn = gr.Button("Process Turn")
            call_chat = gr.Chatbot(label="Call Transcript", height=260)
            call_chat_state = gr.State([])
            call_voice_out = gr.Audio(label="Pebble Voice Reply", autoplay=True)
            gr.Markdown("If browser says no microphone found, use uploaded clips until mic permission is enabled.")

        with gr.TabItem("Telegram Bot"):
            gr.Markdown("### Telegram Bot Voice Settings")
            gr.Markdown("Select a bot and configure its voice settings.")
            
            # Bot selector
            bot_names = get_bot_names()
            selected_bot = gr.Dropdown(
                label="Select Bot",
                choices=bot_names,
                value=bot_names[0] if bot_names else "Pebble",
            )
            
            # Load settings for selected bot
            def _get_bot_voice_settings(bot_name: str) -> Tuple[str, str]:
                """Get voice settings for a specific bot."""
                bot_config = get_bot_config(bot_name)
                if bot_config:
                    return (
                        bot_config.get("voice_name", "Pebble"),
                        bot_config.get("voice_mode", "Text Only"),
                    )
                return "Pebble", "Text Only"
            
            initial_voice, initial_mode = _get_bot_voice_settings(bot_names[0] if bot_names else "Pebble")
            
            with gr.Row():
                telegram_voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=VOICE_NAMES,
                    value=initial_voice,
                )
                telegram_mode_radio = gr.Radio(
                    label="Reply Mode",
                    choices=["Text Only", "Text + Voice"],
                    value=initial_mode,
                )
            
            save_telegram_btn = gr.Button("Save Settings", variant="primary")
            telegram_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("**Current Settings:**")
            current_settings_display = gr.Textbox(
                label="",
                value=f"Voice: {initial_voice} | Mode: {initial_mode}",
                interactive=False,
            )
            
            # Bot selector change handler - load bot's settings
            def _on_bot_select(bot_name: str) -> Tuple[str, str, str]:
                voice, mode = _get_bot_voice_settings(bot_name)
                display = f"Voice: {voice} | Mode: {mode}"
                return voice, mode, display
            
            selected_bot.change(
                _on_bot_select,
                inputs=[selected_bot],
                outputs=[telegram_voice_dropdown, telegram_mode_radio, current_settings_display],
            )
            
            # Save handler - save to bots_config.json
            def _save_bot_voice_settings(bot_name: str, voice: str, mode: str) -> Tuple[str, str]:
                """Save voice settings for a specific bot."""
                update_bot(bot_name, voice_name=voice, voice_mode=mode)
                display = f"Voice: {voice} | Mode: {mode}"
                return f"✅ Saved! Bot: {bot_name}, Voice: {voice}, Mode: {mode}", display
            
            save_telegram_btn.click(
                _save_bot_voice_settings,
                inputs=[selected_bot, telegram_voice_dropdown, telegram_mode_radio],
                outputs=[telegram_status, current_settings_display],
            )

    outputs = [brain_status, senses_status, bot_status, brain_log, senses_log, bot_log]
    demo.load(refresh, outputs=outputs)
    refresh_btn.click(refresh, outputs=outputs)
    start_brain_btn.click(start_brain, outputs=outputs)
    stop_brain_btn.click(stop_brain, outputs=outputs)
    start_senses_btn.click(start_senses, outputs=outputs)
    stop_senses_btn.click(stop_senses, outputs=outputs)
    start_bot_btn.click(start_bot, outputs=outputs)
    stop_bot_btn.click(stop_bot, outputs=outputs)
    start_all_btn.click(start_all, outputs=outputs)
    stop_all_btn.click(stop_all, outputs=outputs)

    send_btn.click(
        send_text,
        inputs=[profile, chat_in, state, voice_toggle],
        outputs=[chat, status, chat_in, voice_out],
    ).then(lambda c: c, inputs=[chat], outputs=[state])

    send_audio_btn.click(
        send_uploaded_audio,
        inputs=[profile, audio_in, state, voice_toggle],
        outputs=[chat, status, voice_out],
    ).then(lambda c: c, inputs=[chat], outputs=[state])

    clear_btn.click(clear_chat, outputs=[chat, status, chat_in, voice_out]).then(lambda: [], outputs=[state])

    call_on.change(toggle_call, inputs=[call_on], outputs=[call_state, call_status])
    calibrate_btn.click(calibrate_noise, inputs=[noise_clip], outputs=[noise_threshold, call_status])
    process_turn_btn.click(
        process_call_turn,
        inputs=[call_profile, call_on, noise_threshold, call_audio_in, call_chat_state],
        outputs=[call_chat, call_state, call_status, call_voice_out],
    ).then(lambda c: c, inputs=[call_chat], outputs=[call_chat_state])

    # Telegram Bot tab handlers
    save_telegram_btn.click(
        _save_telegram_settings,
        inputs=[telegram_voice_dropdown, telegram_mode_radio],
        outputs=[telegram_status, current_settings_display],
    )

    # Profile change handler - load chat history when profile changes
    profile.change(
        _load_history_on_start,
        inputs=[profile],
        outputs=[chat],
    ).then(lambda c: c, inputs=[chat], outputs=[state])


demo.launch(server_name="127.0.0.1", server_port=7862)
