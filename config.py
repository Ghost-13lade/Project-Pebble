"""
Configuration management for Conscious Pebble.
Loads settings from environment variables and .env file in data directory.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

# Try to import dotenv, but don't crash if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ENV_FILE = DATA_DIR / ".env"

# =============================================================================
# PROVIDER PRESETS
# =============================================================================

PROVIDER_PRESETS = {
    "Local MLX": {
        "base_url": "http://localhost:8080/v1",
        "api_key": "local-dev-key",
        "model": "local-model",
        "requires_mlx": True,
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "openrouter/optimus-alpha",
        "requires_mlx": False,
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "requires_mlx": False,
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key": "",
        "model": "mistral-small-latest",
        "requires_mlx": False,
    },
    "LM Studio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "local-model",
        "requires_mlx": False,
    },
    "Ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3.2",
        "requires_mlx": False,
    },
}

# =============================================================================
# ENVIRONMENT LOADING
# =============================================================================

_env_loaded = False


def load_env() -> None:
    """Load environment variables from .env file."""
    global _env_loaded
    if DOTENV_AVAILABLE and ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    _env_loaded = True


def reload_env() -> None:
    """Force reload environment variables from .env file."""
    global _env_loaded
    _env_loaded = False
    load_env()


# Load on module import
load_env()

# =============================================================================
# CONFIG GETTERS
# =============================================================================

def get_config(key: str, default: str = "") -> str:
    """Get a configuration value from environment."""
    return os.getenv(key, default)


def get_env_path() -> Path:
    """Get the path to the .env file."""
    return ENV_FILE


def get_provider() -> str:
    """Get the current LLM provider."""
    return get_config("LLM_PROVIDER", "Local MLX")


def get_api_key() -> str:
    """Get the API key for the LLM provider."""
    return get_config("OPENAI_API_KEY", "YOUR_KEY_HERE")


def get_base_url() -> str:
    """Get the base URL for the LLM API."""
    return get_config("OPENAI_BASE_URL", "http://localhost:8080/v1")


def get_model() -> str:
    """Get the model name for the LLM."""
    return get_config("OPENAI_MODEL", "local-model")


def get_telegram_token() -> str:
    """Get the Telegram bot token."""
    return get_config("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")


def get_allowed_user_id() -> str:
    """Get the allowed Telegram user ID."""
    return get_config("ALLOWED_USER_ID", "")


def get_senses_base_url() -> str:
    """Get the senses service base URL."""
    return get_config("SENSES_BASE_URL", "http://localhost:8081")


def get_mlx_model_path() -> str:
    """Get the MLX model path for local inference."""
    return get_config("MLX_MODEL_PATH", "mlx-community/Llama-3.2-3B-Instruct-4bit")


def get_mlx_kv_bits() -> str:
    """Get the MLX KV cache quantization bits."""
    return get_config("MLX_KV_BITS", "4")


def get_whisper_model() -> str:
    """Get the Whisper model for STT."""
    return get_config("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")


def get_kokoro_model() -> str:
    """Get the Kokoro model for TTS."""
    return get_config("KOKORO_MODEL", "mlx-community/Kokoro-82M-bf16")


def get_elevenlabs_api_key() -> str:
    """Get the ElevenLabs API key for cloud TTS."""
    return get_config("ELEVENLABS_API_KEY", "")


def get_elevenlabs_voice_id() -> str:
    """Get the ElevenLabs voice ID (default: Rachel)."""
    return get_config("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice


def get_tts_provider() -> str:
    """Get the TTS provider: 'local' (MLX/Kokoro), 'elevenlabs' (Cloud), 'openai', or 'none'."""
    return get_config("TTS_PROVIDER", "local")


def get_stt_provider() -> str:
    """Get the STT provider: 'local' (MLX Whisper), 'groq', or 'openai'."""
    return get_config("STT_PROVIDER", "local")


def get_groq_api_key() -> str:
    """Get the Groq API key for cloud Whisper STT."""
    return get_config("GROQ_API_KEY", "")


def get_openai_tts_key() -> str:
    """Get the OpenAI API key for TTS (falls back to main OPENAI_API_KEY)."""
    return get_config("OPENAI_TTS_API_KEY", "") or get_api_key()


def get_openai_tts_voice() -> str:
    """Get the OpenAI TTS voice (alloy, echo, fable, onyx, nova, shimmer)."""
    return get_config("OPENAI_TTS_VOICE", "alloy")


def get_web_search_enabled() -> bool:
    """Check if web search is enabled."""
    return get_config("WEB_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes")


# =============================================================================
# CONFIG SETTERS
# =============================================================================

def save_env_value(key: str, value: str) -> None:
    """Save a single value to the .env file."""
    # Read existing content
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().strip().split("\n")
    
    # Update or add the key
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"{key}={value}")
    
    # Write back
    ENV_FILE.write_text("\n".join(new_lines) + "\n")
    
    # Update current environment
    os.environ[key] = value


def save_config(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    telegram_token: Optional[str] = None,
    allowed_user_id: Optional[str] = None,
    mlx_model_path: Optional[str] = None,
    mlx_kv_bits: Optional[str] = None,
) -> None:
    """Save multiple configuration values to the .env file."""
    if provider is not None:
        save_env_value("LLM_PROVIDER", provider)
    if api_key is not None:
        save_env_value("OPENAI_API_KEY", api_key)
    if base_url is not None:
        save_env_value("OPENAI_BASE_URL", base_url)
    if model is not None:
        save_env_value("OPENAI_MODEL", model)
    if telegram_token is not None:
        save_env_value("TELEGRAM_BOT_TOKEN", telegram_token)
    if allowed_user_id is not None:
        save_env_value("ALLOWED_USER_ID", allowed_user_id)
    if mlx_model_path is not None:
        save_env_value("MLX_MODEL_PATH", mlx_model_path)
    if mlx_kv_bits is not None:
        save_env_value("MLX_KV_BITS", mlx_kv_bits)
    
    # Reload environment
    reload_env()


def apply_provider_preset(provider: str) -> Dict[str, str]:
    """Apply a provider preset and return the values."""
    preset = PROVIDER_PRESETS.get(provider, {})
    if preset:
        save_config(
            provider=provider,
            base_url=preset.get("base_url", ""),
            model=preset.get("model", ""),
        )
        if preset.get("api_key"):
            # Don't overwrite existing API key with preset placeholder
            pass
    return preset


# =============================================================================
# MLX MODEL MANAGEMENT
# =============================================================================

def scan_models_directory(root_dir: str) -> List[str]:
    """
    Scan a directory for valid MLX model subfolders.
    A valid model folder contains config.json or config.yaml.
    Returns a list of folder names (not full paths).
    """
    if not root_dir or not Path(root_dir).exists():
        return []
    
    models = []
    root_path = Path(root_dir)
    
    try:
        for item in root_path.iterdir():
            if item.is_dir():
                # Check for config files (indicates valid model)
                config_json = item / "config.json"
                config_yaml = item / "config.yaml"
                config_yml = item / "config.yml"
                
                if config_json.exists() or config_yaml.exists() or config_yml.exists():
                    models.append(item.name)
    except PermissionError:
        pass
    
    return sorted(models)


def get_mlx_models_root() -> str:
    """Get the MLX models root directory."""
    return get_config("MLX_MODELS_ROOT", "")


def get_mlx_selected_model() -> str:
    """Get the selected MLX model name (subfolder name or HF ID)."""
    return get_config("MLX_SELECTED_MODEL", "")


def get_mlx_context_size() -> str:
    """Get the MLX context size (max_kv_size)."""
    return get_config("MLX_CONTEXT_SIZE", "8192")


def save_mlx_config(
    models_root: Optional[str] = None,
    selected_model: Optional[str] = None,
    kv_bits: Optional[str] = None,
    context_size: Optional[str] = None,
) -> None:
    """Save MLX configuration to .env file."""
    if models_root is not None:
        save_env_value("MLX_MODELS_ROOT", models_root)
    if selected_model is not None:
        save_env_value("MLX_SELECTED_MODEL", selected_model)
        # Also update MLX_MODEL_PATH for backwards compatibility
        if models_root and selected_model:
            # Check if it's a local path or HF ID
            if Path(models_root).exists():
                model_path = str(Path(models_root) / selected_model)
            else:
                model_path = selected_model  # Assume HF ID
            save_env_value("MLX_MODEL_PATH", model_path)
    if kv_bits is not None:
        save_env_value("MLX_KV_BITS", kv_bits)
    if context_size is not None:
        save_env_value("MLX_CONTEXT_SIZE", context_size)
    
    reload_env()


def resolve_mlx_model_path() -> str:
    """
    Resolve the full MLX model path.
    Priority:
    1. MLX_MODELS_ROOT + MLX_SELECTED_MODEL (local folder)
    2. MLX_MODEL_PATH (direct path or HF ID)
    3. Default HF ID
    """
    models_root = get_mlx_models_root()
    selected_model = get_mlx_selected_model()
    
    if models_root and selected_model:
        local_path = Path(models_root) / selected_model
        if local_path.exists():
            return str(local_path)
    
    # Fall back to direct path or HF ID
    return get_mlx_model_path()


def fetch_provider_models(provider: str, api_key: str = "") -> List[str]:
    """
    Fetch available models from a provider's API.
    Returns a list of model names.
    """
    import httpx
    
    preset = PROVIDER_PRESETS.get(provider, {})
    base_url = preset.get("base_url", "")
    
    if not base_url:
        return []
    
    # Use provided API key or fall back to saved key
    key = api_key or get_api_key()
    if not key:
        return []
    
    headers = {"Authorization": f"Bearer {key}"}
    
    # Provider-specific endpoints
    endpoints = {
        "OpenAI": "/models",
        "OpenRouter": "/models",
        "Mistral": "/models",
        "LM Studio": "/models",
        "Ollama": "/api/tags",  # Different endpoint for Ollama
    }
    
    endpoint = endpoints.get(provider, "/models")
    url = base_url.rstrip("/") + endpoint
    
    try:
        if provider == "Ollama":
            # Ollama uses different response format
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        else:
            # OpenAI-compatible APIs
            response = httpx.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", "") for m in data.get("data", [])]
    except Exception as e:
        print(f"[Config] Failed to fetch models for {provider}: {e}")
    
    return []


# =============================================================================
# LEGACY COMPATIBILITY - Module-level constants for existing code
# =============================================================================

# These are evaluated at import time. For dynamic config, use the getter functions.
BOT_TOKEN = get_config("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT_TOKEN = get_config("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
OPENAI_BASE_URL = get_base_url()
OPENAI_MODEL = get_model()
OPENAI_API_KEY = get_api_key()
ALLOWED_USER_ID = get_allowed_user_id()

SENSES_BASE_URL = get_senses_base_url()
SENSES_HEAR_PATH = get_config("SENSES_HEAR_PATH", "/hear")
SENSES_SPEAK_PATH = get_config("SENSES_SPEAK_PATH", "/speak")

WHISPER_MODEL = get_whisper_model()
KOKORO_MODEL = get_kokoro_model()

MLX_MODEL_PATH = get_mlx_model_path()
MLX_KV_BITS = get_mlx_kv_bits()