import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import librosa
import numpy as np

from config import (
    KOKORO_MODEL,
    SENSES_BASE_URL,
    SENSES_HEAR_PATH,
    SENSES_SPEAK_PATH,
    WHISPER_MODEL,
    get_elevenlabs_api_key,
    get_elevenlabs_voice_id,
    get_tts_provider,
    get_stt_provider,
    get_groq_api_key,
    get_openai_tts_key,
    get_openai_tts_voice,
)


BASE_DIR = Path(__file__).resolve().parent
VOICE_CONFIG_FILE = BASE_DIR / "true_voices.json"


def load_voice_configs() -> List[Dict[str, Any]]:
    if not VOICE_CONFIG_FILE.exists():
        return []
    try:
        payload = json.loads(VOICE_CONFIG_FILE.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def default_voice_name() -> str:
    configs = load_voice_configs()
    if configs:
        name = str(configs[0].get("name", "")).strip()
        if name:
            return name
    return "Pebble"


def resolve_voice_preset(active_voice_name: str) -> Dict[str, Any]:
    fallback = {
        "name": active_voice_name or default_voice_name(),
        "voice": "af_heart",
        "speed": 1.0,
        "pitch_shift": 0,
    }
    for cfg in load_voice_configs():
        if str(cfg.get("name", "")).strip() == active_voice_name:
            return {
                "name": str(cfg.get("name", fallback["name"])),
                "voice": str(cfg.get("voice", fallback["voice"])),
                "speed": float(cfg.get("speed", fallback["speed"])),
                "pitch_shift": float(cfg.get("pitch_shift", fallback["pitch_shift"])),
            }
    return fallback


def _build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def transcribe_audio_file(audio_path: str) -> Optional[str]:
    """
    Local STT using Senses service (MLX Whisper).
    Only works on Mac with Apple Silicon.
    """
    print("[Voice] Using Local MLX Whisper (requires Mac/MLX)")
    try:
        endpoint = _build_url(SENSES_BASE_URL, SENSES_HEAR_PATH)
        with open(audio_path, "rb") as f:
            files = {
                "file": (
                    Path(audio_path).name,
                    f.read(),
                    "application/octet-stream",
                )
            }
        data = {"model": WHISPER_MODEL}

        with httpx.Client(timeout=180.0) as client:
            response = client.post(endpoint, files=files, data=data)
            response.raise_for_status()
            payload = response.json()

        text = str(payload.get("text", "")).strip()
        if text:
            print(f"[Voice Engine] Local STT success: '{text[:50]}...'")
        return text or None
    except Exception as e:
        print(f"[Voice Engine] Transcription server call failed: {e}")
        return None


def transcribe_audio_groq(audio_path: str) -> Optional[str]:
    """
    Cloud STT using Groq's Whisper API.
    Fast, free tier available. Works on Windows/Linux/Mac.
    """
    api_key = get_groq_api_key()
    if not api_key:
        print("[Voice Engine] Groq STT selected but no API key configured!")
        print("[Voice Engine] Set GROQ_API_KEY in your .env file or Settings tab")
        return None
    
    print("[Voice] Using Groq Whisper API (Cloud)...")
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (Path(audio_path).name, audio_file, "audio/wav"),
            }
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
            }
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, files=files, data=data)
                
                if response.status_code == 401:
                    print("[Voice Engine] Groq: Invalid API key")
                    return None
                
                if response.status_code == 413:
                    print("[Voice Engine] Groq: File too large (max 25MB)")
                    return None
                
                response.raise_for_status()
                payload = response.json()
        
        text = str(payload.get("text", "")).strip()
        if text:
            print(f"[Voice Engine] Groq STT success: '{text[:50]}...'")
        return text or None
        
    except httpx.TimeoutException:
        print("[Voice Engine] Groq: Request timed out")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Voice Engine] Groq: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[Voice Engine] Groq STT error: {e}")
        return None


def transcribe_audio_openai(audio_path: str) -> Optional[str]:
    """
    Cloud STT using OpenAI's Whisper API.
    Works on Windows/Linux/Mac.
    """
    from config import get_api_key
    api_key = get_api_key()
    
    if not api_key or api_key == "YOUR_KEY_HERE":
        print("[Voice Engine] OpenAI STT selected but no API key configured!")
        return None
    
    print("[Voice] Using OpenAI Whisper API (Cloud)...")
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (Path(audio_path).name, audio_file, "audio/wav"),
            }
            data = {
                "model": "whisper-1",
                "response_format": "json",
            }
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, files=files, data=data)
                
                if response.status_code == 401:
                    print("[Voice Engine] OpenAI: Invalid API key")
                    return None
                
                response.raise_for_status()
                payload = response.json()
        
        text = str(payload.get("text", "")).strip()
        if text:
            print(f"[Voice Engine] OpenAI STT success: '{text[:50]}...'")
        return text or None
        
    except httpx.TimeoutException:
        print("[Voice Engine] OpenAI: Request timed out")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Voice Engine] OpenAI: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[Voice Engine] OpenAI STT error: {e}")
        return None


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Main STT dispatcher - checks provider and routes accordingly.
    
    Provider options:
    - 'local': Use local Senses service (Mac only, requires MLX)
    - 'groq': Use Groq Whisper API (Cloud, fast, free tier)
    - 'openai': Use OpenAI Whisper API (Cloud)
    """
    provider = get_stt_provider()
    print(f"[Voice] STT Provider: {provider}")
    
    if provider == "groq":
        return transcribe_audio_groq(audio_path)
    elif provider == "openai":
        return transcribe_audio_openai(audio_path)
    else:  # local (default)
        return transcribe_audio_file(audio_path)


def extract_emotion_tag(audio_path: str) -> str:
    """Lightweight acoustic heuristic tag for prompt enrichment."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return "neutral"
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        duration = float(len(y) / sr)

        if rms < 0.02:
            return "soft, low-energy"
        if rms > 0.08 and zcr > 0.09:
            return "energetic, excited"
        if duration > 8 and rms < 0.05:
            return "reflective, calm"
        return "warm, neutral"
    except Exception:
        return "neutral"


def _emotion_speed_multiplier(emotion: str) -> float:
    value = str(emotion or "neutral").strip().lower()
    if value in {"excited", "happy", "angry"}:
        return 1.15
    if value in {"sad", "tired", "thoughtful"}:
        return 0.85
    return 1.0


def generate_audio_elevenlabs(
    text: str,
    api_key: str,
    voice_id: str,
) -> Optional[io.BytesIO]:
    """
    Generate audio using ElevenLabs API.
    Bypasses local Senses service entirely - works on Windows/Linux.
    """
    if not text.strip():
        print("[Voice Engine] ElevenLabs: Empty text, skipping")
        return None
    
    if not api_key:
        print("[Voice Engine] ElevenLabs: No API key configured")
        return None
    
    print(f"[Voice] Using ElevenLabs API (Cloud) for voice: {voice_id}")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",  # Fast, good quality
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
            if response.status_code == 401:
                print("[Voice Engine] ElevenLabs: Invalid API key")
                return None
            
            if response.status_code == 422:
                print(f"[Voice Engine] ElevenLabs: Invalid voice ID '{voice_id}'")
                return None
            
            response.raise_for_status()
            
            # ElevenLabs returns audio directly
            content_type = response.headers.get("content-type", "")
            if "audio" in content_type:
                print(f"[Voice Engine] ElevenLabs: Success! Received {len(response.content)} bytes")
                return io.BytesIO(response.content)
            else:
                print(f"[Voice Engine] ElevenLabs: Unexpected content type: {content_type}")
                return None
                
    except httpx.TimeoutException:
        print("[Voice Engine] ElevenLabs: Request timed out")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Voice Engine] ElevenLabs: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[Voice Engine] ElevenLabs: Unexpected error: {e}")
        return None


def generate_audio_openai(
    text: str,
    api_key: str,
    voice: str = "alloy",
) -> Optional[io.BytesIO]:
    """
    Generate audio using OpenAI TTS API.
    Works on Windows/Linux/Mac.
    
    Voice options: alloy, echo, fable, onyx, nova, shimmer
    """
    if not text.strip():
        print("[Voice Engine] OpenAI TTS: Empty text, skipping")
        return None
    
    if not api_key:
        print("[Voice Engine] OpenAI TTS: No API key configured")
        return None
    
    print(f"[Voice] Using OpenAI TTS API (Cloud) for voice: {voice}")
    
    url = "https://api.openai.com/v1/audio/speech"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "tts-1",  # Fast, good quality (tts-1-hd for higher quality)
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
            if response.status_code == 401:
                print("[Voice Engine] OpenAI TTS: Invalid API key")
                return None
            
            if response.status_code == 400:
                print(f"[Voice Engine] OpenAI TTS: Invalid voice '{voice}'")
                print("[Voice Engine] Valid voices: alloy, echo, fable, onyx, nova, shimmer")
                return None
            
            response.raise_for_status()
            
            # OpenAI returns audio directly
            content_type = response.headers.get("content-type", "")
            if "audio" in content_type or len(response.content) > 1000:
                print(f"[Voice Engine] OpenAI TTS: Success! Received {len(response.content)} bytes")
                return io.BytesIO(response.content)
            else:
                print(f"[Voice Engine] OpenAI TTS: Unexpected response")
                return None
                
    except httpx.TimeoutException:
        print("[Voice Engine] OpenAI TTS: Request timed out")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Voice Engine] OpenAI TTS: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[Voice Engine] OpenAI TTS: Unexpected error: {e}")
        return None


def synthesize_voice(
    text: str,
    active_voice_name: str = "Pebble",
    detected_emotion: str = "neutral",
) -> Optional[io.BytesIO]:
    """
    Main TTS dispatcher - checks provider and routes accordingly.
    
    Provider options:
    - 'elevenlabs': Use ElevenLabs cloud API (Windows/Linux compatible)
    - 'openai': Use OpenAI TTS API (Windows/Linux/Mac)
    - 'local': Use local Senses service (Mac only, requires MLX)
    - 'none': Disable voice entirely
    """
    if not text.strip():
        return None
    
    provider = get_tts_provider()
    print(f"[Voice] TTS Provider: {provider}")
    
    # ElevenLabs Cloud (works on Windows/Linux)
    if provider == "elevenlabs":
        api_key = get_elevenlabs_api_key()
        voice_id = get_elevenlabs_voice_id()
        
        if not api_key:
            print("[Voice Engine] ElevenLabs selected but no API key configured!")
            print("[Voice Engine] Set ELEVENLABS_API_KEY in your .env file or Settings tab")
            return None
        
        return generate_audio_elevenlabs(text, api_key, voice_id)
    
    # OpenAI TTS Cloud (works on Windows/Linux/Mac)
    elif provider == "openai":
        api_key = get_openai_tts_key()
        voice = get_openai_tts_voice()
        
        if not api_key or api_key == "YOUR_KEY_HERE":
            print("[Voice Engine] OpenAI TTS selected but no API key configured!")
            print("[Voice Engine] Set OPENAI_API_KEY in your .env file or Settings tab")
            return None
        
        return generate_audio_openai(text, api_key, voice)
    
    # Local Senses Service (Mac only)
    elif provider == "local":
        print("[Voice] Using Local Senses Service (requires Mac/MLX)")
        return synthesize_voice_bytes_local(text, active_voice_name, detected_emotion)
    
    # Voice disabled
    else:
        print("[Voice] TTS Provider set to 'none' - voice disabled")
        return None


def synthesize_voice_bytes_local(
    text: str,
    active_voice_name: str,
    detected_emotion: str = "neutral",
) -> Optional[io.BytesIO]:
    """
    Local TTS using Senses service (Kokoro/MLX).
    Only works on Mac with Apple Silicon.
    """
    if not text.strip():
        return None

    cfg = resolve_voice_preset(active_voice_name)
    voice = str(cfg.get("voice", "af_heart"))
    base_speed = float(cfg.get("speed", 1.0))
    pitch_shift = float(cfg.get("pitch_shift", 0))
    multiplier = _emotion_speed_multiplier(detected_emotion)
    speed = max(0.5, min(1.5, base_speed * multiplier))

    try:
        endpoint = _build_url(SENSES_BASE_URL, SENSES_SPEAK_PATH)
        payload = {
            "model": KOKORO_MODEL,
            "text": text,
            "voice": voice,
            "speed": speed,
            "pitch": pitch_shift,
        }

        with httpx.Client(timeout=180.0) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body = response.json()
            # Support {"audio_base64":"..."} style APIs if used.
            b64 = body.get("audio_base64")
            if isinstance(b64, str) and b64:
                import base64

                return io.BytesIO(base64.b64decode(b64))
            print("[Voice Engine] Kokoro server returned JSON without audio bytes")
            return None

        print(f"[Voice Engine] Local TTS success! Received {len(response.content)} bytes")
        return io.BytesIO(response.content)
    except Exception as e:
        print(f"[Voice Engine] Local TTS server call failed: {e}")
        print("[Voice Engine] Hint: Is the Senses service running on port 8081?")
        return None


# Backward compatibility wrapper
def synthesize_voice_bytes(
    text: str,
    active_voice_name: str,
    detected_emotion: str = "neutral",
) -> Optional[io.BytesIO]:
    """
    Legacy wrapper - routes through synthesize_voice.
    Maintains backward compatibility with existing code.
    """
    return synthesize_voice(text, active_voice_name, detected_emotion)
