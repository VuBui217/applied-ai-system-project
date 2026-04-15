"""
Agentic Loop for VibeFinder 2.0

Coordinates intent parsing, RAG retrieval, scoring, self-evaluation,
and re-ranking into a single observable pipeline.

Observable steps (stored in AgentResult.steps):
  1. parse_intent       — Gemini extracts UserProfile from plain English
  2. rag_retrieve       — semantic search returns candidate songs
  3. score_candidates   — score_song() ranks the candidates
  4. self_evaluate      — checks for genre-lock or energy mismatch
  5. rerank             — fires only when genre-lock is detected
  6. diversity_check    — ensures top-5 spans at least 2 genres

Public API
----------
run_agent(query, songs, k=5) -> AgentResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.intent_parser import parse_intent
from src.retriever import build_index, retrieve
from src.recommender import score_song
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One observable reasoning step produced by the agent."""
    name: str
    detail: str


@dataclass
class Recommendation:
    """A single song recommendation with score, confidence, and explanation."""
    song: Dict
    score: float
    confidence: float   # score / MAX_SCORE, clamped to [0, 1]
    explanation: str


@dataclass
class AgentResult:
    """Everything the agent produced for one query."""
    query: str
    profile: Dict
    recommendations: List[Recommendation]
    steps: List[AgentStep]
    genre_lock_detected: bool = False
    genre_lock_corrected: bool = False


# ---------------------------------------------------------------------------
# Scoring constants (must match recommender.py)
# ---------------------------------------------------------------------------
# With the weight-shift experiment applied:
#   energy × 2  (max 2.0)
#   acoustic    (max 1.0)
#   genre bonus (max 1.0)
#   mood bonus  (max 1.0)
MAX_SCORE = 5.0
GENRE_BONUS = 1.0
MOOD_BONUS  = 1.0

# Genre-lock thresholds
_ENERGY_DEVIATION_THRESHOLD = 0.30   # top song's |target - actual| > this
_BONUS_DOMINANCE_THRESHOLD  = 1.5    # genre+mood bonus >= this AND energy is poor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_all(profile: Dict, candidates: List[Dict]) -> List[Tuple[Dict, float]]:
    """Score every candidate with the standard scoring function."""
    return sorted(
        [(s, score_song(profile, s)) for s in candidates],
        key=lambda x: x[1],
        reverse=True,
    )


def _build_explanation(profile: Dict, song: Dict, score: float) -> str:
    reasons = []
    if song["genre"] == profile["favorite_genre"]:
        reasons.append(f"genre matches ({song['genre']})")
    if song["mood"] == profile["favorite_mood"]:
        reasons.append(f"mood matches ({song['mood']})")
    energy_dev = abs(profile["target_energy"] - song["energy"])
    if energy_dev <= 0.15:
        reasons.append(
            f"energy is close ({song['energy']:.2f} vs target {profile['target_energy']:.2f})"
        )
    elif energy_dev <= 0.30:
        reasons.append(
            f"energy is near ({song['energy']:.2f} vs target {profile['target_energy']:.2f})"
        )
    if not reasons:
        reasons.append("closest overall match available")
    return f"Score {score:.2f}/5.00 — " + ", ".join(reasons) + "."


def _detect_genre_lock(profile: Dict, scored: List[Tuple[Dict, float]]) -> Tuple[bool, str]:
    """
    Return (is_locked, reason_string).

    Genre-lock is flagged when ALL of:
    - Top song earned the genre bonus (genre matches)
    - Top song has a meaningful energy mismatch (deviation > threshold)
    - The genre+mood bonuses account for >= _BONUS_DOMINANCE_THRESHOLD of its score
    """
    if not scored:
        return False, ""

    top_song, top_score = scored[0]
    got_genre_bonus = int(top_song["genre"] == profile["favorite_genre"]) * GENRE_BONUS
    got_mood_bonus  = int(top_song["mood"]  == profile["favorite_mood"])  * MOOD_BONUS
    total_bonus     = got_genre_bonus + got_mood_bonus
    energy_dev      = abs(profile["target_energy"] - top_song["energy"])

    if (
        got_genre_bonus > 0
        and energy_dev > _ENERGY_DEVIATION_THRESHOLD
        and total_bonus >= _BONUS_DOMINANCE_THRESHOLD
    ):
        reason = (
            f"'{top_song['title']}' won mainly on bonuses ({total_bonus:.1f} pts) "
            f"but has energy mismatch of {energy_dev:.2f} "
            f"(target {profile['target_energy']:.2f}, actual {top_song['energy']:.2f})"
        )
        return True, reason

    return False, ""


def _rerank_without_genre_lock(
    profile: Dict, candidates: List[Dict]
) -> List[Tuple[Dict, float]]:
    """
    Re-score candidates with genre bonus zeroed out so energy/acoustic/mood
    drive the ranking instead.
    """
    no_genre_profile = {**profile, "favorite_genre": "__none__"}
    return _score_all(no_genre_profile, candidates)


def _ensure_diversity(
    scored: List[Tuple[Dict, float]], k: int
) -> List[Tuple[Dict, float]]:
    """
    Return up to k recommendations ensuring at least 2 distinct genres
    appear in the list (if the pool is large enough).
    Preserves score order within each genre slot.
    """
    seen_genres: set = set()
    result: List[Tuple[Dict, float]] = []
    remainder: List[Tuple[Dict, float]] = []

    for song, score in scored:
        if len(result) >= k:
            break
        if song["genre"] not in seen_genres or len(seen_genres) >= 2:
            result.append((song, score))
            seen_genres.add(song["genre"])
        else:
            remainder.append((song, score))

    # Back-fill remaining slots with next-best scores
    for item in remainder:
        if len(result) >= k:
            break
        result.append(item)

    return result[:k]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(query: str, songs: List[Dict], k: int = 5) -> AgentResult:
    """
    Run the full agentic recommendation pipeline.

    Parameters
    ----------
    query : str   Plain-English music request from the user.
    songs : list  Full song catalog (list of dicts from load_songs).
    k     : int   Number of final recommendations to return (default 5).

    Returns
    -------
    AgentResult with recommendations, observable steps, and bias flags.
    """
    # ------------------------------------------------------------------
    # Guardrail: validate user input before touching any API
    # ------------------------------------------------------------------
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty. Please describe the music you want.")
    if len(query) < 3:
        raise ValueError(f"Query too short ({len(query)} chars). Please be more descriptive.")
    if len(query) > 500:
        log.warning("Query exceeds 500 chars — truncating to 500.")
        query = query[:500]

    log.info("Agent started for query: %r", query)
    steps: List[AgentStep] = []

    # ------------------------------------------------------------------
    # Step 1: Parse intent
    # ------------------------------------------------------------------
    profile = parse_intent(query)
    step1_detail = (
        f"Gemini extracted → genre={profile['favorite_genre']}, "
        f"mood={profile['favorite_mood']}, "
        f"energy={profile['target_energy']}, "
        f"acoustic={profile['likes_acoustic']}. "
        f"Reasoning: {profile.get('reasoning', 'n/a')}"
    )
    steps.append(AgentStep(name="parse_intent", detail=step1_detail))
    log.debug("[step1/parse_intent] %s", step1_detail)

    # ------------------------------------------------------------------
    # Step 2: RAG retrieval
    # ------------------------------------------------------------------
    n_retrieve = min(20, len(songs))  # retrieve more than k to allow re-ranking
    candidates_raw = retrieve(query, k=n_retrieve)
    # retrieve() returns song dicts with extra keys (similarity, description)
    # score_song() only uses the standard song keys, so this is safe
    step2_detail = (
        f"Semantic search returned {len(candidates_raw)} candidates. "
        f"Top match: '{candidates_raw[0]['title']}' "
        f"(similarity={candidates_raw[0]['similarity']:.3f})"
    )
    steps.append(AgentStep(name="rag_retrieve", detail=step2_detail))
    log.debug("[step2/rag_retrieve] %s", step2_detail)

    # ------------------------------------------------------------------
    # Step 3: Score candidates
    # ------------------------------------------------------------------
    scored = _score_all(profile, candidates_raw)
    top_title  = scored[0][0]["title"]
    top_score  = scored[0][1]
    step3_detail = (
        f"Scored {len(scored)} candidates. "
        f"Current #1: '{top_title}' with score {top_score:.2f}/5.00."
    )
    steps.append(AgentStep(name="score_candidates", detail=step3_detail))
    log.debug("[step3/score_candidates] %s", step3_detail)

    # ------------------------------------------------------------------
    # Step 4: Self-evaluate for genre-lock
    # ------------------------------------------------------------------
    genre_locked, lock_reason = _detect_genre_lock(profile, scored)
    genre_lock_corrected = False
    step4_detail = (
        f"Genre-lock detected: {genre_locked}. "
        + (lock_reason if genre_locked else "Ranking looks fair.")
    )
    steps.append(AgentStep(name="self_evaluate", detail=step4_detail))
    if genre_locked:
        log.warning("[step4/self_evaluate] Genre-lock detected — %s", lock_reason)
    else:
        log.debug("[step4/self_evaluate] %s", step4_detail)

    # ------------------------------------------------------------------
    # Step 5: Re-rank if genre-locked
    # ------------------------------------------------------------------
    if genre_locked:
        scored = _rerank_without_genre_lock(profile, candidates_raw)
        new_top = scored[0][0]["title"]
        step5_detail = (
            f"Genre bonus zeroed. New #1: '{new_top}' "
            f"(score {scored[0][1]:.2f}/5.00). "
            f"Re-ranking prioritises energy and mood fit."
        )
        steps.append(AgentStep(name="rerank", detail=step5_detail))
        log.info("[step5/rerank] %s", step5_detail)
        genre_lock_corrected = True

    # ------------------------------------------------------------------
    # Step 6: Diversity check
    # ------------------------------------------------------------------
    diverse_scored = _ensure_diversity(scored, k)
    genres_in_top = {s["genre"] for s, _ in diverse_scored}
    step6_detail = (
        f"Final top-{k} spans {len(genres_in_top)} genre(s): "
        f"{', '.join(sorted(genres_in_top))}."
    )
    steps.append(AgentStep(name="diversity_check", detail=step6_detail))
    log.debug("[step6/diversity_check] %s", step6_detail)

    # ------------------------------------------------------------------
    # Build final Recommendation objects
    # ------------------------------------------------------------------
    recommendations = [
        Recommendation(
            song=song,
            score=round(score, 2),
            confidence=round(min(score / MAX_SCORE, 1.0), 3),
            explanation=_build_explanation(profile, song, score),
        )
        for song, score in diverse_scored
    ]

    log.info(
        "Agent complete — %d recommendations. Genre-lock: detected=%s corrected=%s",
        len(recommendations), genre_locked, genre_lock_corrected,
    )
    return AgentResult(
        query=query,
        profile=profile,
        recommendations=recommendations,
        steps=steps,
        genre_lock_detected=genre_locked,
        genre_lock_corrected=genre_lock_corrected,
    )


# ---------------------------------------------------------------------------
# Pretty printer (used by main.py and the CLI smoke-test)
# ---------------------------------------------------------------------------

def print_agent_result(result: AgentResult) -> None:
    """Print a formatted agent result to stdout."""
    WIDTH = 60
    print("\n" + "=" * WIDTH)
    print(f"  QUERY: {result.query}")
    print("=" * WIDTH)

    print("\n[Agent Reasoning]")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. [{step.name}] {step.detail}")

    if result.genre_lock_detected:
        tag = "CORRECTED" if result.genre_lock_corrected else "DETECTED"
        print(f"\n  ⚠  Genre-lock {tag} — re-ranking applied.")

    print(f"\n[Profile] genre={result.profile['favorite_genre']}  "
          f"mood={result.profile['favorite_mood']}  "
          f"energy={result.profile['target_energy']}  "
          f"acoustic={result.profile['likes_acoustic']}")

    print("\n[Recommendations]")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"\n  #{i}  {rec.song['title']}  —  {rec.song['artist']}")
        print(f"       Genre: {rec.song['genre']}  |  "
              f"Mood: {rec.song['mood']}  |  "
              f"Energy: {rec.song['energy']:.2f}")
        print(f"       Score: {rec.score}/5.00  |  "
              f"Confidence: {rec.confidence:.0%}")
        print(f"       Why:   {rec.explanation}")
    print()


# ---------------------------------------------------------------------------
# CLI smoke-test  (python -m src.agent)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from src.recommender import load_songs

    print("Loading catalog and building index...")
    all_songs = load_songs("data/songs.csv")
    build_index(all_songs)
    print(f"Ready. {len(all_songs)} songs indexed.\n")

    test_queries = [
        "something chill to study to late at night",
        "I want high energy music for my morning run",
        "give me something sad and slow for a rainy afternoon",
        # Adversarial: should trigger genre-lock detection
        "I need fast and intense blues music",
    ]

    for q in test_queries:
        result = run_agent(q, all_songs, k=5)
        print_agent_result(result)
        time.sleep(15)   # respect free-tier rate limit between calls
