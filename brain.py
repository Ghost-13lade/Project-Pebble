import calendar
import json
import os
import random
import re
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI

from db import get_user_profile
from prompts import (
    load_dream_prompt,
    load_loop_followup_prompt,
    load_reminiscence_prompt,
    load_soul_prompt,
    load_spontaneous_prompt,
)
from emotional_core import EmotionalCore
from memory_engine import MemoryEngine

# Pattern definitions
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
EMOTION_TAG_PATTERN = re.compile(r"\[emotion:\s*(\w+)\]", re.IGNORECASE)
NEEDS_CHECK_PATTERN = re.compile(r"\[NEEDS CHECK:.*?\]", re.DOTALL | re.IGNORECASE)
EOT_TOKEN_PATTERN = re.compile(r"<\|eot_id\|>", re.IGNORECASE)
EOS_TOKEN_PATTERN = re.compile(r"</s>", re.IGNORECASE)


class Brain:
    def __init__(
        self,
        model: str = "hermes-4-70b",
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        memory_engine: MemoryEngine | None = None,
        emotional_core: EmotionalCore | None = None,
    ) -> None:
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "local-dev-key"),
            timeout=300.0,
        )
        self.memory_engine = memory_engine or MemoryEngine()
        self.emotional_core = emotional_core or EmotionalCore()
        # Load prompts from files at init
        self._soul_prompt = load_soul_prompt()

    def _get_weather_for_user(self, user_id: str) -> str:
        """Get weather based on user's location."""
        try:
            profile = get_user_profile(user_id)
            location = profile.get("location", "").strip()
            if not location:
                return "Unknown location (User hasn't told me where they live yet)"
            from tools import get_current_weather
            weather_data = get_current_weather(location)
            return f"{weather_data} in {location}"
        except Exception:
            return "Unknown weather"

    def _parse_timestamp(self, value: str | None) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone().replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _format_time_since_last_interaction(
        self,
        history: List[Dict[str, str]],
        now: datetime,
    ) -> str:
        latest_timestamp: Optional[datetime] = None
        for item in history:
            parsed = self._parse_timestamp(item.get("created_at"))
            if parsed and (latest_timestamp is None or parsed > latest_timestamp):
                latest_timestamp = parsed
        if latest_timestamp is None:
            return "0 minutes"
        delta = now - latest_timestamp
        seconds = max(int(delta.total_seconds()), 0)
        if seconds >= 86400:
            days = max(seconds // 86400, 1)
            return f"{days} day" + ("s" if days != 1 else "")
        if seconds >= 3600:
            hours = max(seconds // 3600, 1)
            return f"{hours} hour" + ("s" if hours != 1 else "")
        minutes = seconds // 60
        return f"{minutes} minute" + ("s" if minutes != 1 else "")

    def _build_messages(
        self,
        history: List[Dict[str, str]],
        persona: str,
        user_profile: str,
        bot_name: str = "Pebble",
        user_name: str = "you",
        retrieved_context: str = "",
        current_weather: str = "Unknown",
        relationship_status: str = "We are getting to know each other.",
        delivery_mode: str = "text",
        user_length_hint: str = "medium",
    ) -> List[Dict[str, str]]:
        now = datetime.now()
        time_since_last_interaction = self._format_time_since_last_interaction(history, now)
        current_date = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
        emotional_state = self.emotional_core.load()
        current_mood = str(emotional_state.get("current_mood", "warm and attentive"))
        attachment_level = float(emotional_state.get("attachment_level", 5.0))
        pending_loops = self.emotional_core.get_pending_loops()
        if pending_loops:
            pending_open_loops = "\n".join(
                f"- {str(loop.get('topic', '')).strip()} (expected: {str(loop.get('expected_time', 'soon')).strip() or 'soon'})"
                for loop in pending_loops
                if str(loop.get("topic", "")).strip()
            )
            if not pending_open_loops.strip():
                pending_open_loops = "None"
        else:
            pending_open_loops = "None"
        memory_parts: List[str] = []
        if retrieved_context:
            memory_parts.append(f"[Relevant Memories ONLY if directly tied to current message]:\n{retrieved_context}")
        if user_profile:
            memory_parts.append(f"[Pebble's Inner Notes on Us]:\n{user_profile}")
        retrieved_memories = "\n\n".join(memory_parts) if memory_parts else "None"
        # Use soul prompt from file with dynamic names
        rendered_base_prompt = self._soul_prompt.format(
            bot_name=bot_name,
            user_name=user_name,
            current_date=current_date,
            time_since_last_interaction=time_since_last_interaction,
            current_weather=current_weather,
            current_mood=current_mood,
            attachment_level=f"{attachment_level:.1f}",
            relationship_status=relationship_status,
            pending_open_loops=pending_open_loops,
            retrieved_memories=retrieved_memories,
            delivery_mode=delivery_mode,
            user_length_hint=user_length_hint,
        )
        # Force the model to prioritize the new persona over old history
        style_enforcer = (
            "\n\n[CRITICAL INSTRUCTION: IGNORE PAST TONE]\n"
            "The user may have just switched your 'Mode'. "
            "If the recent conversation history has a different tone (e.g., bossy, formal), **DROP IT IMMEDIATELY**.\n"
            "You must align 100% with the [Persona Prompt] defined above.\n"
            "Do not repeat the user's text. Respond directly to them."
        )

        system_message = (
            f"[Base Soul Prompt]\n{rendered_base_prompt}\n\n"
            f"[Persona Prompt]\n{persona}\n"
            f"{style_enforcer}"
        )

        # === DEBUG LOGGING: BRAIN ACTIVITY ===
        print(f"[BROOK BRAIN] Generating response...")
        print(f"[BROOK BRAIN] Weather Context Injected: '{current_weather}'")
        print(f"[BROOK BRAIN] System Prompt Size: {len(system_message)} chars")
        # =====================================
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role and content is not None:
                messages.append({"role": str(role), "content": str(content)})
        if str(delivery_mode).strip().lower() == "voice":
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "[CURRENT MODE: VOICE CALL - STRICT]\n"
                        "You are speaking out loud naturally.\n"
                        "- Plain text only — NO emojis, NO markdown, NO asterisks.\n"
                        "- NO lists or numbered items.\n"
                        "- VARY sentences — don't repeat same structure.\n"
                        "- Casual spoken flow: fillers like 'umm', 'you know', trailing thoughts.\n"
                        "- Short and breathable — like real conversation."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "[CURRENT MODE: TEXT CHAT]\n"
                        "You are texting casually.\n"
                        "- Mostly lowercase.\n"
                        "- Emojis okay but sparse (only when truly feeling it, max 1-2).\n"
                        "- No markdown."
                    ),
                }
            )
        return messages

    def _strip_thoughts(self, content: str) -> Tuple[str, List[str]]:
        thoughts = [match.strip() for match in THINK_TAG_PATTERN.findall(content)]
        cleaned = THINK_TAG_PATTERN.sub("", content).strip()
        return cleaned, thoughts

    def _clean_model_output(self, content: str) -> str:
        cleaned = content or ""
        cleaned = EOT_TOKEN_PATTERN.sub("", cleaned)
        cleaned = EOS_TOKEN_PATTERN.sub("", cleaned)
        cleaned = NEEDS_CHECK_PATTERN.sub("", cleaned)
        cleaned = THINK_TAG_PATTERN.sub("", cleaned)
        user_cutoff = cleaned.find("\n\nUser:")
        if user_cutoff != -1:
            cleaned = cleaned[:user_cutoff]
        return cleaned.strip()

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.8) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=["<|im_end|>", "<|eot_id|>"],
            temperature=temperature,
            timeout=300.0,
        )
        message = completion.choices[0].message
        return message.content or message.reasoning or ""

    def _extract_emotion(self, text: str) -> Tuple[str, str]:
        if not text:
            return "", "neutral"
        match = EMOTION_TAG_PATTERN.search(text)
        if not match:
            return text.strip(), "neutral"
        emotion = str(match.group(1)).strip().lower() or "neutral"
        cleaned = EMOTION_TAG_PATTERN.sub("", text, count=1).strip()
        return cleaned, emotion

    def generate_response(
        self,
        history: List[Dict[str, str]],
        persona: str,
        user_profile: str,
        bot_name: str = "Pebble",
        user_name: str = "you",
        retrieved_context: str = "",
        current_weather: str = "Unknown",
        user_id: str = "",
        relationship_status: str = "We are getting to know each other.",
        delivery_mode: str = "text",
        user_length_hint: str = "medium",
    ) -> Tuple[str, str]:
        # === WEB SEARCH CHECK ===
        web_search_results = ""
        latest_user_text = ""
        for item in reversed(history):
            if str(item.get("role", "")).lower() == "user":
                latest_user_text = str(item.get("content", "")).strip()
                break
        
        if latest_user_text:
            from config import get_web_search_enabled
            from tools_search import needs_web_search, extract_search_query, search_web
            
            if get_web_search_enabled() and needs_web_search(latest_user_text):
                search_query = extract_search_query(latest_user_text)
                web_search_results = search_web(search_query)
                if web_search_results:
                    # Inject web results into retrieved_context
                    if retrieved_context:
                        retrieved_context = f"{retrieved_context}\n\n[Web Search Results]:\n{web_search_results}"
                    else:
                        retrieved_context = f"[Web Search Results]:\n{web_search_results}"
        
        # === MEMORY RETRIEVAL WITH VERBOSE LOGGING ===
        print(f"[Memory] Starting context retrieval for user: {user_id}")
        print(f"[Context] Loaded {len(history)} recent messages from session")
        
        memory_context = retrieved_context
        if user_id and not memory_context:
            if not latest_user_text:
                for item in reversed(history):
                    if str(item.get("role", "")).lower() == "user":
                        latest_user_text = str(item.get("content", "")).strip()
                        break
            if latest_user_text:
                print(f"[Memory] Querying vector DB with: '{latest_user_text[:100]}...'")
                memory_context = self.memory_engine.retrieve_relevant_context(
                    query=latest_user_text,
                    user_id=user_id,
                    k=5,  # Increased from default 3 for better context
                )
                # Log retrieval results
                has_events = "[Past Related Events]:" in memory_context and "None" not in memory_context.split("[Past Related Events]:")[1].split("[Relevant Facts]:")[0]
                has_facts = "[Relevant Facts]:" in memory_context and "None" not in memory_context.split("[Relevant Facts]:")[1] if "[Relevant Facts]:" in memory_context else False
                print(f"[Memory] Retrieved relevant memories - Events: {has_events}, Facts: {has_facts}")
                
                # Add web search results to memory context if we have them
                if web_search_results and memory_context:
                    memory_context = f"{memory_context}\n\n{web_search_results}"
        else:
            print(f"[Memory] Using provided context (length: {len(memory_context)} chars)")
        messages = self._build_messages(
            history=history,
            persona=persona,
            user_profile=user_profile,
            bot_name=bot_name,
            user_name=user_name,
            retrieved_context=memory_context,
            current_weather=current_weather,
            relationship_status=relationship_status,
            delivery_mode=delivery_mode,
            user_length_hint=user_length_hint,
        )
        raw_output = ""
        retries = 0
        while retries < 2:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.85,
                    presence_penalty=0.3,
                    frequency_penalty=0.6,
                    max_tokens=2500,
                    stop=["<|im_end|>", "<|eot_id|>"],
                    timeout=300.0,
                )
                message = completion.choices[0].message
                raw_output = message.content or message.reasoning or ""
                print(f"[DEBUG] Pulled from reasoning: {bool(message.reasoning and not message.content)}")
                print(f"[DEBUG] Raw output start: '{raw_output[:200] if raw_output else 'EMPTY'}'")
                if not raw_output.strip():
                    retries += 1
                    print(f"[Brain Warning] Empty raw output on attempt {retries}. Retrying with adjusted temperature.")
                    continue
                break
            except Exception as e:
                retries += 1
                print(f"[Brain Error] Completion failed on attempt {retries}: {e}")
        thinks = [match.strip() for match in THINK_TAG_PATTERN.findall(raw_output)]
        clean_output = self._clean_model_output(raw_output)
        if thinks:
            print("\n[Brain Debug | <think> traces]")
            for idx, thought in enumerate(thinks, start=1):
                print(f"{idx}. {thought}")
        clean_output, detected_emotion = self._extract_emotion(clean_output)
        if clean_output.strip():
            final_reply = clean_output.strip()
        else:
            if thinks:
                last_thought = " ".join(thinks[-1].split()).strip()
                final_reply = f"mm... {last_thought.lower()}\n\nyou know, thinking about that makes me feel closer to you ❤️\nwhat's on your mind?"
            else:
                final_reply = "hey babe... got caught up in my thoughts for a sec\nsay that again?"
                detected_emotion = "neutral"
        print(f"[Output Debug] Raw len: {len(raw_output)}, Thinks: {len(thinks)}, Clean len: {len(clean_output)}, Final len: {len(final_reply)}")
        return final_reply, (detected_emotion or "neutral")

    def detect_reminder(self, text: str) -> Optional[Dict[str, str]]:
        lowered = text.lower()
        recurring_cues = ("every day", "daily", "every night")
        if not any(keyword in lowered for keyword in ("remind", "alarm", "alert", *recurring_cues)):
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract reminder intent from user text. "
                    "Return ONLY JSON with this exact schema: "
                    '{"type": "recurring|one_off", "interval": "daily or null", '
                    '"time": "HH:MM", "task": "task_string"}. '
                    "If user says 'every day', 'daily', or 'every night', "
                    "set type='recurring' and interval='daily'. "
                    "If unclear, return {}."
                ),
            },
            {"role": "user", "content": text},
        ]
        raw = self._chat(messages=messages, temperature=0.2).strip()
        try:
            parsed: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None
        reminder_type = str(parsed.get("type", "one_off")).strip().lower() or "one_off"
        interval_value = parsed.get("interval")
        interval = str(interval_value).strip().lower() if interval_value is not None else ""
        time_value = str(parsed.get("time", "")).strip()
        task_value = str(parsed.get("task", "")).strip()
        if any(cue in lowered for cue in recurring_cues):
            reminder_type = "recurring"
            interval = "daily"
        if reminder_type not in {"one_off", "recurring"}:
            reminder_type = "one_off"
        if reminder_type == "recurring" and interval != "daily":
            interval = "daily"
        if not time_value or not task_value:
            return None
        return {"type": reminder_type, "interval": interval if interval else None, "time": time_value, "task": task_value}

    def extract_location(self, text: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "Check if this text contains a user stating their location/city. If yes, return ONLY the city/state as a plain string. If no, return ONLY NONE."},
            {"role": "user", "content": text},
        ]
        raw = self._chat(messages=messages, temperature=0.0).strip()
        cleaned = self._clean_model_output(raw).strip().strip('"').strip("'")
        if not cleaned or cleaned.upper() == "NONE":
            return None
        return cleaned

    def dream_process(self, chat_logs: List[Dict[str, str]]) -> str:
        logs_blob = "\n".join(f"[{item.get('created_at', '')}] {item.get('role', 'unknown')}: {item.get('content', '')}" for item in chat_logs)
        messages = [
            {"role": "system", "content": "Analyze these chat logs. Summarize the key events, the user's emotional state, and any new facts learned. Output a concise summary."},
            {"role": "user", "content": logs_blob},
        ]
        return self._chat(messages=messages, temperature=0.4).strip()

    def run_dream_cycle(self, chat_logs: List[Dict[str, str]], user_id: str = "default", date: str | date_type | None = None) -> str:
        if not chat_logs:
            return ""
        logs_blob = "\n".join(f"[{item.get('created_at', '')}] {item.get('role', 'unknown')}: {item.get('content', '')}" for item in chat_logs)
        # Use dream prompt from file
        dream_prompt = load_dream_prompt()
        messages = [{"role": "system", "content": dream_prompt}, {"role": "user", "content": logs_blob}]
        raw = self._chat(messages=messages, temperature=0.4).strip()
        diary_entry = ""
        attachment_delta = 0.0
        mood = "warm and attentive"
        open_loops: List[Dict[str, str]] = []
        try:
            parsed = json.loads(raw)
            diary_entry = str(parsed.get("diary_entry", "")).strip()
            attachment_delta = float(parsed.get("attachment_delta", 0.0))
            mood = str(parsed.get("mood", mood)).strip() or mood
            loops_raw = parsed.get("open_loops", [])
            if isinstance(loops_raw, list):
                for item in loops_raw:
                    if isinstance(item, dict):
                        topic = str(item.get("topic", "")).strip()
                        expected_time = str(item.get("expected_time", "soon")).strip() or "soon"
                        if topic:
                            open_loops.append({"topic": topic, "expected_time": expected_time})
        except (json.JSONDecodeError, ValueError, TypeError):
            diary_entry = self.dream_process(chat_logs)
        if not diary_entry:
            diary_entry = self.dream_process(chat_logs)
        day_value = date.isoformat() if isinstance(date, date_type) else (date or datetime.now().date().isoformat())
        self.memory_engine.archive_day(summary_text=diary_entry, date=day_value, user_id=user_id)
        previous_state = self.emotional_core.load()
        previous_attachment = float(previous_state.get("attachment_level", 5.0))
        updated_state = self.emotional_core.update(mood=mood, attachment_delta=attachment_delta)
        new_attachment = float(updated_state.get("attachment_level", previous_attachment))
        for loop in open_loops:
            self.emotional_core.add_loop(topic=str(loop.get("topic", "")).strip(), time_hint=str(loop.get("expected_time", "soon")).strip() or "soon")
        if int(new_attachment) > int(previous_attachment) and user_id and user_id != "default":
            relationship_messages = [
                {"role": "system", "content": f"Our attachment level just reached {int(new_attachment)}. Define our relationship status in 1 sentence based on our history. Return plain text only."},
                {"role": "user", "content": logs_blob},
            ]
            relationship_status = self._chat(relationship_messages, temperature=0.4).strip()
            relationship_status = self._clean_model_output(relationship_status) or "We are getting to know each other."
            try:
                from db import get_user_profile, upsert_user_profile
                profile = get_user_profile(user_id)
                upsert_user_profile(user_id=user_id, summary=profile.get("summary", ""), emotional_notes=profile.get("emotional_notes", ""), day_summary=profile.get("day_summary", ""), location=profile.get("location", ""), relationship_status=relationship_status)
            except Exception:
                pass
        return diary_entry

    def _is_loop_due_or_close(self, expected_time: str) -> bool:
        hint = (expected_time or "").strip().lower()
        if not hint:
            return False
        immediate_tokens = ("today", "tonight", "this evening", "this afternoon", "this morning", "now", "soon", "in an hour", "later today", "tomorrow", "tmr")
        if any(token in hint for token in immediate_tokens):
            return True
        weekday_map = {day.lower(): idx for idx, day in enumerate(calendar.day_name)}
        weekday_map.update({day.lower()[:3]: idx for idx, day in enumerate(calendar.day_name)})
        for token, day_idx in weekday_map.items():
            if token in hint:
                now = datetime.now()
                delta = (day_idx - now.weekday()) % 7
                return delta <= 1
        return False

    def get_due_open_loop(self) -> Optional[Dict[str, str]]:
        loops = self.emotional_core.get_pending_loops()
        for loop in loops:
            if self._is_loop_due_or_close(str(loop.get("expected_time", ""))):
                return loop
        return None

    def decide_to_message(self, last_interaction_time: datetime | str | None, attachment_level: float) -> bool:
        if self.get_due_open_loop():
            return True
        now = datetime.now()
        if now.hour >= 23 or now.hour < 8:
            return False
        last_time: Optional[datetime] = None
        if isinstance(last_interaction_time, datetime):
            last_time = last_interaction_time
        elif isinstance(last_interaction_time, str):
            last_time = self._parse_timestamp(last_interaction_time)
        if last_time is None:
            return False
        gap_hours = max((now - last_time).total_seconds(), 0.0) / 3600.0
        if gap_hours < 4:
            return False
        probability = 0.05 + (float(attachment_level) * 0.02)
        if gap_hours > 24:
            probability += 0.20
        probability = max(0.0, min(0.95, probability))
        return random.random() < probability

    def generate_loop_followup(self, topic: str, expected_time: str = "soon") -> str:
        # Use loop followup prompt from file
        prompt_template = load_loop_followup_prompt()
        prompt = prompt_template.format(topic=topic, expected_time=expected_time)
        messages = [{"role": "system", "content": prompt}]
        raw = self._chat(messages=messages, temperature=0.8)
        return self._clean_model_output(raw)

    def generate_spontaneous_thought(self, gap: str, mood: str, weather: str) -> str:
        # Use spontaneous prompt from file
        prompt_template = load_spontaneous_prompt()
        prompt = prompt_template.format(gap=gap, mood=mood, weather=weather)
        messages = [{"role": "system", "content": prompt}]
        raw = self._chat(messages=messages, temperature=0.8)
        return self._clean_model_output(raw)

    def generate_reminiscence_thought(self, random_memory_summary: str) -> str:
        # Use reminiscence prompt from file
        prompt_template = load_reminiscence_prompt()
        prompt = prompt_template.format(memory_summary=random_memory_summary)
        messages = [{"role": "system", "content": prompt}]
        raw = self._chat(messages=messages, temperature=0.7)
        return self._clean_model_output(raw)

    def generate_custom_persona_prompt(self, description: str) -> str:
        messages = [
            {"role": "system", "content": "You are a persona prompt engineer. Create a concise but expressive system prompt for a local companion AI. Include voice texture, mood sync behavior, and imperfection cues."},
            {"role": "user", "content": f"Persona description: {description}"},
        ]
        return self._chat(messages=messages, temperature=0.7).strip()

    def consolidate_profile_from_logs(self, day_logs: List[Dict[str, str]], previous_summary: str, previous_emotional_notes: str) -> Dict[str, str]:
        logs_blob = "\n".join(f"[{item.get('created_at', '')}] {item['role']}: {item['content']}" for item in day_logs)
        psychologist_prompt = "You are a careful memory consolidation psychologist for an AI companion. Read the chat logs and update user memory in JSON. Return ONLY valid JSON with keys: summary, emotional_notes, day_summary."
        messages = [{"role": "system", "content": psychologist_prompt}, {"role": "user", "content": f"Previous summary:\n{previous_summary}\n\nPrevious emotional notes:\n{previous_emotional_notes}\n\nToday's logs:\n{logs_blob}"}]
        raw = self._chat(messages=messages, temperature=0.3).strip()
        try:
            parsed = json.loads(raw)
            return {"summary": parsed.get("summary", previous_summary), "emotional_notes": parsed.get("emotional_notes", previous_emotional_notes), "day_summary": parsed.get("day_summary", "")}
        except json.JSONDecodeError:
            return {"summary": previous_summary, "emotional_notes": previous_emotional_notes, "day_summary": "Unable to parse dream summary JSON."}

    def extract_facts_from_summary(self, summary_text: str) -> List[str]:
        if not summary_text.strip():
            return []
        messages = [{"role": "system", "content": "Extract concrete user facts and goals from the summary. Return ONLY valid JSON as {\"facts\": [\"...\"]}."}, {"role": "user", "content": summary_text}]
        raw = self._chat(messages=messages, temperature=0.2).strip()
        try:
            data = json.loads(raw)
            facts = data.get("facts", [])
            if isinstance(facts, list):
                return [str(item).strip() for item in facts if str(item).strip()]
        except json.JSONDecodeError:
            return []
        return []

    def extract_names_from_text(self, text: str) -> Optional[Dict[str, str]]:
        """Extract user's name and what they want to call the bot from their message."""
        messages = [
            {"role": "system", "content": (
                "Extract names from the user's message. "
                "Return ONLY valid JSON with this exact schema: "
                "{\"user_name\": \"name for the user\", \"bot_name\": \"name the user wants to call you\"}. "
                "Example: {\"user_name\": \"Yuri\", \"bot_name\": \"Pebble\"}. "
                "If unclear, return {}."
            )},
            {"role": "user", "content": text},
        ]
        raw = self._chat(messages=messages, temperature=0.2).strip()
        try:
            parsed = json.loads(raw)
            if parsed and "user_name" in parsed and "bot_name" in parsed:
                return {
                    "user_name": str(parsed.get("user_name", "")).strip(),
                    "bot_name": str(parsed.get("bot_name", "")).strip(),
                }
        except json.JSONDecodeError:
            pass
        return None
