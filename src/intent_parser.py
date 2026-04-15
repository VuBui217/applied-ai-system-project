"""
Intent Parser for VibeFinder 2.0

Sends the user's plain-English music request to Gemini and gets back a
structured UserProfile dict that the existing score_song() function understands.

Public API
----------
parse_intent(query: str) -> dict
    Returns a UserProfile-compatible dict plus a "reasoning" key explaining
    how Gemini interpreted the request.

    Example return value:
    {
        "favorite_genre":  "lofi",
        "favorite_mood":   "focused",
        "target_energy":   0.35,
        "likes_acoustic":  True,
        "reasoning":       "User wants calm background music for studying..."
    }
"""

import os
import json
import time
from typing import Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client (created once)
# ---------------------------------------------------------------------------
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-2.5-flash"   # free-tier model, fast and capable

# ---------------------------------------------------------------------------
# Valid catalog values — Gemini must pick from these so scores are meaningful
# ---------------------------------------------------------------------------
VALID_GENRES = [
    "pop", "lofi", "rock", "metal", "jazz", "ambient", "synthwave",
    "indie pop", "r&b", "country", "folk", "edm", "blues", "reggae",
    "hip-hop", "classical", "soul", "punk", "latin", "trap",
]

VALID_MOODS = [
    "happy", "chill", "intense", "relaxed", "romantic", "angry",
    "nostalgic", "melancholic", "euphoric", "sad", "laid-back", "peaceful",
    "confident", "moody", "focused", "warm", "dreamy", "excited",
]

_SYSTEM_PROMPT = f"""You are a music preference parser for VibeFinder, a music recommender.

Given a user's natural-language description of what music they want, extract their
preferences as a JSON object with EXACTLY these fields:

{{
  "favorite_genre":  "<one genre from the list below>",
  "favorite_mood":   "<one mood from the list below>",
  "target_energy":   <number 0.0–1.0>,
  "likes_acoustic":  <true or false>,
  "reasoning":       "<one sentence explaining your interpretation>"
}}

Available genres  : {", ".join(VALID_GENRES)}
Available moods   : {", ".join(VALID_MOODS)}

Energy scale:
  0.0–0.25  very quiet / calm / meditative
  0.25–0.45 low energy, relaxed background feel
  0.45–0.65 moderate — present but not overwhelming
  0.65–0.80 upbeat and active
  0.80–1.0  intense, loud, high-octane

likes_acoustic:
  true  → user wants organic, natural, unplugged sound (guitar, piano, voice)
  false → user wants electronic, produced, or heavily processed sound

Rules:
- You MUST return only the JSON object — no markdown, no extra text.
- favorite_genre and favorite_mood MUST be values from the lists above.
- If the user does not mention a specific genre, infer the closest one from context.
- If the user gives contradictory signals (e.g. "calm metal"), weight mood and energy
  over genre.
"""


def parse_intent(query: str) -> Dict:
    """
    Parse a plain-English music request into a structured UserProfile dict.

    Parameters
    ----------
    query : str
        What the user typed, e.g. "I want something to hype me up at the gym"

    Returns
    -------
    dict with keys: favorite_genre, favorite_mood, target_energy,
                    likes_acoustic, reasoning

    Raises
    ------
    ValueError  if Gemini returns something that cannot be parsed or contains
                invalid field values.
    """
    # Retry up to 3 times on transient server errors (503) or rate limits (429)
    last_error = None
    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=_MODEL,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,    # low temp = more deterministic parsing
                    max_output_tokens=1024,
                ),
            )
            break   # success — exit retry loop
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "503" in err_str or "429" in err_str:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                time.sleep(wait)
            else:
                raise   # non-retryable error — surface immediately
    else:
        raise RuntimeError(
            f"Gemini API failed after 3 attempts: {last_error}"
        )

    raw = response.text.strip()

    # Strip occasional markdown fences or preamble text before the JSON object
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    # Find the first '{' in case Gemini prepends a sentence
    brace_idx = raw.find("{")
    if brace_idx > 0:
        raw = raw[brace_idx:]

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON output: {raw!r}"
        ) from exc

    _validate(profile)
    return profile


def _validate(profile: Dict) -> None:
    """Raise ValueError if any required field is missing or out of range."""
    required = {"favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"Gemini response missing fields: {missing}")

    if profile["favorite_genre"] not in VALID_GENRES:
        raise ValueError(
            f"Unknown genre {profile['favorite_genre']!r}. "
            f"Must be one of: {VALID_GENRES}"
        )

    if profile["favorite_mood"] not in VALID_MOODS:
        raise ValueError(
            f"Unknown mood {profile['favorite_mood']!r}. "
            f"Must be one of: {VALID_MOODS}"
        )

    energy = profile["target_energy"]
    if not isinstance(energy, (int, float)) or not (0.0 <= float(energy) <= 1.0):
        raise ValueError(
            f"target_energy must be a float between 0.0 and 1.0, got {energy!r}"
        )

    if not isinstance(profile["likes_acoustic"], bool):
        raise ValueError(
            f"likes_acoustic must be true or false, got {profile['likes_acoustic']!r}"
        )

    # Normalise — ensure energy is a Python float
    profile["target_energy"] = float(profile["target_energy"])


# ---------------------------------------------------------------------------
# CLI smoke-test  (python -m src.intent_parser)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    test_queries = [
        "I want something to hype me up at the gym",
        "chill background music while I study late at night",
        "sad and slow, kind of rainy day vibes",
        "something happy and danceable for a party",
        "dark and heavy, I'm in an angry mood",
        "I need peaceful acoustic guitar music to help me sleep",
        "upbeat latin beats to cook dinner to",
    ]

    print("=" * 60)
    print("Intent Parser — Gemini Smoke Test")
    print("=" * 60)
    for q in test_queries:
        print(f"\nQuery   : {q}")
        try:
            result = parse_intent(q)
            print(f"Genre   : {result['favorite_genre']}")
            print(f"Mood    : {result['favorite_mood']}")
            print(f"Energy  : {result['target_energy']}")
            print(f"Acoustic: {result['likes_acoustic']}")
            print(f"Reason  : {result['reasoning']}")
        except ValueError as e:
            print(f"ERROR   : {e}")
        time.sleep(13)   # stay within 5 req/min free-tier limit
