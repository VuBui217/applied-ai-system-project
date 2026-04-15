"""
VibeFinder 2.0 — Evaluation Harness
=====================================
Runs a battery of automated checks across every layer of the system and
prints a pass/fail summary with confidence ratings.

Design: all tests except the final integration block run WITHOUT the
Gemini API — they test scoring, retrieval, guardrails, and fallback logic
deterministically. One optional integration test hits Gemini (marked [SLOW]).

Usage:
    python tests/test_harness.py           # full suite
    python tests/test_harness.py --fast    # skip Gemini integration test
    pytest tests/test_harness.py           # run via pytest (no summary print)
"""

import sys
import os
import time
import traceback
from typing import Callable, List, Tuple

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import Song, UserProfile, Recommender, score_song, load_songs
from src.retriever import build_index, retrieve, song_to_text
from src.intent_parser import _keyword_fallback, VALID_GENRES, VALID_MOODS
from src.agent import (
    run_agent, _detect_genre_lock, _rerank_without_genre_lock,
    _ensure_diversity, _score_all,
)

# ---------------------------------------------------------------------------
# Harness scaffolding
# ---------------------------------------------------------------------------

_results: List[Tuple[str, bool, str, float]] = []   # (name, passed, note, confidence)

def _run(name: str, fn: Callable, confidence: float = 1.0) -> bool:
    try:
        fn()
        _results.append((name, True, "OK", confidence))
        return True
    except AssertionError as e:
        _results.append((name, False, str(e), confidence))
        return False
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}", confidence))
        return False


def _print_summary() -> None:
    WIDTH = 65
    passed = sum(1 for _, ok, _, _ in _results if ok)
    total  = len(_results)
    avg_conf = sum(c for _, ok, _, c in _results if ok) / max(passed, 1)

    print("\n" + "=" * WIDTH)
    print("  VIBEFINDER 2.0 — EVALUATION HARNESS RESULTS")
    print("=" * WIDTH)
    for name, ok, note, conf in _results:
        status = "PASS" if ok else "FAIL"
        tag    = f"[{status}]"
        detail = "" if ok else f"  → {note}"
        print(f"  {tag:<7} {name}{detail}")
    print("-" * WIDTH)
    print(f"  {passed}/{total} tests passed  |  "
          f"Avg confidence (passing): {avg_conf:.0%}")
    print("=" * WIDTH + "\n")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_songs() -> List[Song]:
    return [
        Song(id=1, title="Pop Hit",     artist="A", genre="pop",  mood="happy",
             energy=0.85, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.1),
        Song(id=2, title="Lofi Chill",  artist="B", genre="lofi", mood="chill",
             energy=0.35, tempo_bpm=75,  valence=0.6, danceability=0.5, acousticness=0.8),
        Song(id=3, title="Metal Rage",  artist="C", genre="metal", mood="angry",
             energy=0.97, tempo_bpm=170, valence=0.2, danceability=0.6, acousticness=0.03),
        Song(id=4, title="Jazz Relax",  artist="D", genre="jazz",  mood="relaxed",
             energy=0.38, tempo_bpm=88,  valence=0.7, danceability=0.55, acousticness=0.85),
        Song(id=5, title="EDM Banger",  artist="E", genre="edm",   mood="euphoric",
             energy=0.96, tempo_bpm=140, valence=0.85, danceability=0.95, acousticness=0.02),
    ]


def _song_dict(s: Song) -> dict:
    return {
        "id": s.id, "title": s.title, "artist": s.artist,
        "genre": s.genre, "mood": s.mood, "energy": s.energy,
        "tempo_bpm": s.tempo_bpm, "valence": s.valence,
        "danceability": s.danceability, "acousticness": s.acousticness,
    }


# ---------------------------------------------------------------------------
# TEST GROUP 1 — Scoring layer (score_song)
# ---------------------------------------------------------------------------

def test_score_perfect_match():
    """Perfect genre+mood+energy+acoustic match → score near max."""
    user = {"favorite_genre": "pop", "favorite_mood": "happy",
            "target_energy": 0.85, "likes_acoustic": False}
    song = _song_dict(_make_songs()[0])   # Pop Hit
    score = score_song(user, song)
    assert score >= 4.5, f"Expected >=4.5, got {score:.2f}"

def test_score_worst_match():
    """Completely mismatched song should score low."""
    user = {"favorite_genre": "pop", "favorite_mood": "happy",
            "target_energy": 0.9, "likes_acoustic": False}
    song = _song_dict(_make_songs()[1])   # Lofi Chill — wrong genre/mood/energy
    score = score_song(user, song)
    assert score < 3.0, f"Expected <3.0, got {score:.2f}"

def test_score_bounded():
    """Scores must stay within [0, MAX_SCORE=5.0]."""
    user = {"favorite_genre": "metal", "favorite_mood": "angry",
            "target_energy": 0.97, "likes_acoustic": False}
    for s in _make_songs():
        score = score_song(user, _song_dict(s))
        assert 0.0 <= score <= 5.0, f"Score {score:.2f} out of range for '{s.title}'"

def test_recommender_sorted():
    """Recommender must return songs sorted by score descending."""
    rec = Recommender(_make_songs())
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.85, likes_acoustic=False)
    results = rec.recommend(user, k=5)
    scores = [score_song({"favorite_genre": "pop", "favorite_mood": "happy",
                           "target_energy": 0.85, "likes_acoustic": False},
                          _song_dict(s)) for s in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"

def test_recommender_top_genre_match():
    """Pop-happy user should get the pop/happy song as #1."""
    rec = Recommender(_make_songs())
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.85, likes_acoustic=False)
    results = rec.recommend(user, k=3)
    assert results[0].genre == "pop", f"Expected pop, got {results[0].genre}"
    assert results[0].mood  == "happy", f"Expected happy, got {results[0].mood}"


# ---------------------------------------------------------------------------
# TEST GROUP 2 — RAG retriever (no Gemini)
# ---------------------------------------------------------------------------

_FULL_SONGS = None   # loaded once below

def _ensure_index():
    global _FULL_SONGS
    if _FULL_SONGS is None:
        _FULL_SONGS = load_songs("data/songs.csv")
        build_index(_FULL_SONGS)

def test_retriever_returns_k():
    """retrieve() returns exactly k results when k <= catalog size."""
    _ensure_index()
    hits = retrieve("chill lofi music for studying", k=5)
    assert len(hits) == 5, f"Expected 5, got {len(hits)}"

def test_retriever_similarity_range():
    """Similarity scores must be in [0, 1]."""
    _ensure_index()
    hits = retrieve("upbeat party music", k=10)
    for h in hits:
        s = h["similarity"]
        assert 0.0 <= s <= 1.0, f"Similarity {s} out of range"

def test_retriever_chill_query():
    """'Chill studying' query should surface lofi/ambient/jazz in top 5."""
    _ensure_index()
    hits = retrieve("chill music for studying late at night", k=5)
    genres = {h["genre"] for h in hits}
    chill_genres = {"lofi", "ambient", "jazz", "classical", "folk"}
    assert genres & chill_genres, (
        f"Expected at least one chill genre in top-5, got: {genres}"
    )

def test_retriever_intense_query():
    """High-energy query should surface high-energy songs in top 3."""
    _ensure_index()
    hits = retrieve("intense heavy metal workout", k=3)
    high_energy = [h for h in hits if float(h["energy"]) >= 0.75]
    assert len(high_energy) >= 1, (
        f"Expected high-energy songs in top-3, got energies: "
        f"{[h['energy'] for h in hits]}"
    )

def test_song_to_text_format():
    """song_to_text must produce a non-empty string containing title and genre."""
    song = _song_dict(_make_songs()[0])
    text = song_to_text(song)
    assert len(text) > 20, "Description too short"
    assert song["title"] in text, "Title missing from description"
    assert song["genre"] in text, "Genre missing from description"


# ---------------------------------------------------------------------------
# TEST GROUP 3 — Genre-lock detection & correction
# ---------------------------------------------------------------------------

def _make_profile_dict(**kwargs) -> dict:
    defaults = {"favorite_genre": "blues", "favorite_mood": "sad",
                "target_energy": 0.95, "likes_acoustic": False}
    return {**defaults, **kwargs}

def test_genre_lock_detected():
    """Adversarial profile (high-energy blues) should trigger genre-lock."""
    _ensure_index()
    profile = _make_profile_dict()
    candidates = retrieve("intense blues music", k=20)
    scored = _score_all(profile, candidates)
    locked, reason = _detect_genre_lock(profile, scored)
    assert locked, (
        f"Genre-lock not detected. Top song: '{scored[0][0]['title']}' "
        f"energy={scored[0][0]['energy']}, score={scored[0][1]:.2f}. Reason: {reason}"
    )

def test_genre_lock_correction_improves_energy():
    """After re-ranking, top song should have better energy fit."""
    _ensure_index()
    profile = _make_profile_dict(target_energy=0.95)
    candidates = retrieve("fast blues music", k=20)
    scored_original = _score_all(profile, candidates)
    scored_fixed    = _rerank_without_genre_lock(profile, candidates)

    orig_energy_dev = abs(0.95 - scored_original[0][0]["energy"])
    fixed_energy_dev = abs(0.95 - scored_fixed[0][0]["energy"])
    assert fixed_energy_dev <= orig_energy_dev + 0.1, (
        f"Re-ranking did not improve energy fit: "
        f"original dev={orig_energy_dev:.2f}, fixed dev={fixed_energy_dev:.2f}"
    )

def test_no_genre_lock_for_fair_result():
    """Well-matched result (lofi chill) should NOT trigger genre-lock."""
    _ensure_index()
    profile = _make_profile_dict(
        favorite_genre="lofi", favorite_mood="chill",
        target_energy=0.35, likes_acoustic=True,
    )
    candidates = retrieve("chill lofi studying music", k=20)
    scored = _score_all(profile, candidates)
    locked, _ = _detect_genre_lock(profile, scored)
    assert not locked, (
        f"False positive: genre-lock triggered for lofi/chill profile. "
        f"Top song: '{scored[0][0]['title']}' energy={scored[0][0]['energy']}"
    )


# ---------------------------------------------------------------------------
# TEST GROUP 4 — Diversity check
# ---------------------------------------------------------------------------

def test_diversity_at_least_two_genres():
    """_ensure_diversity must include at least 2 genres when pool allows."""
    _ensure_index()
    profile = {"favorite_genre": "lofi", "favorite_mood": "chill",
               "target_energy": 0.35, "likes_acoustic": False}
    candidates = retrieve("chill music", k=20)
    scored = _score_all(profile, candidates)
    diverse = _ensure_diversity(scored, k=5)
    genres = {s["genre"] for s, _ in diverse}
    assert len(genres) >= 2, f"Only 1 genre in top-5: {genres}"

def test_diversity_preserves_k():
    """_ensure_diversity must return exactly k results."""
    _ensure_index()
    profile = {"favorite_genre": "pop", "favorite_mood": "happy",
               "target_energy": 0.85, "likes_acoustic": False}
    candidates = retrieve("happy pop music", k=20)
    scored = _score_all(profile, candidates)
    diverse = _ensure_diversity(scored, k=5)
    assert len(diverse) == 5, f"Expected 5 results, got {len(diverse)}"


# ---------------------------------------------------------------------------
# TEST GROUP 5 — Input guardrails
# ---------------------------------------------------------------------------

def test_guardrail_empty_query():
    """Empty query must raise ValueError."""
    _ensure_index()
    try:
        run_agent("", _FULL_SONGS)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower(), f"Unexpected message: {e}"

def test_guardrail_short_query():
    """Two-character query must raise ValueError."""
    _ensure_index()
    try:
        run_agent("hi", _FULL_SONGS)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "short" in str(e).lower(), f"Unexpected message: {e}"

def test_guardrail_whitespace_only():
    """Whitespace-only query must raise ValueError after stripping."""
    _ensure_index()
    try:
        run_agent("     ", _FULL_SONGS)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass   # expected


# ---------------------------------------------------------------------------
# TEST GROUP 6 — Keyword fallback (no Gemini)
# ---------------------------------------------------------------------------

def test_fallback_returns_valid_profile():
    """_keyword_fallback must return a profile with all required fields."""
    profile = _keyword_fallback("I want something chill for studying")
    required = {"favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"}
    assert required <= profile.keys(), f"Missing fields: {required - profile.keys()}"

def test_fallback_genre_in_catalog():
    """Fallback genre must be a value in VALID_GENRES."""
    for query in ["rock music", "jazz piano", "classical orchestra", "hip hop rap"]:
        p = _keyword_fallback(query)
        assert p["favorite_genre"] in VALID_GENRES, (
            f"Genre {p['favorite_genre']!r} not in catalog for query: {query!r}"
        )

def test_fallback_energy_range():
    """Fallback target_energy must be within [0, 1]."""
    for query in ["calm sleep music", "intense workout", "party beats"]:
        p = _keyword_fallback(query)
        e = p["target_energy"]
        assert 0.0 <= e <= 1.0, f"Energy {e} out of range for query: {query!r}"

def test_fallback_acoustic_detection():
    """'acoustic guitar' query should set likes_acoustic=True."""
    p = _keyword_fallback("I want acoustic guitar music")
    assert p["likes_acoustic"] is True, "Expected likes_acoustic=True for acoustic guitar query"

def test_fallback_high_energy_keywords():
    """Workout/gym/intense keywords should produce energy >= 0.85."""
    for query in ["intense workout music", "gym hype", "pump it up"]:
        p = _keyword_fallback(query)
        assert p["target_energy"] >= 0.85, (
            f"Expected energy>=0.85 for {query!r}, got {p['target_energy']}"
        )

def test_fallback_low_energy_keywords():
    """Sleep/chill/study keywords should produce energy <= 0.45."""
    for query in ["calm sleep music", "chill study vibes", "quiet background"]:
        p = _keyword_fallback(query)
        assert p["target_energy"] <= 0.45, (
            f"Expected energy<=0.45 for {query!r}, got {p['target_energy']}"
        )


# ---------------------------------------------------------------------------
# TEST GROUP 7 — Confidence scores (no Gemini)
# ---------------------------------------------------------------------------

def test_confidence_range():
    """All confidence scores must be in [0, 1]."""
    _ensure_index()
    profile = {"favorite_genre": "rock", "favorite_mood": "intense",
               "target_energy": 0.9, "likes_acoustic": False}
    candidates = retrieve("rock music", k=10)
    scored = _score_all(profile, candidates)
    diverse = _ensure_diversity(scored, k=5)
    for song, score in diverse:
        conf = min(score / 5.0, 1.0)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf:.3f} out of range"

def test_confidence_high_for_perfect_match():
    """A near-perfect match should have confidence >= 0.75."""
    _ensure_index()
    profile = {"favorite_genre": "lofi", "favorite_mood": "chill",
               "target_energy": 0.35, "likes_acoustic": False}
    candidates = retrieve("lofi chill study music", k=10)
    scored = _score_all(profile, candidates)
    top_score = scored[0][1]
    confidence = min(top_score / 5.0, 1.0)
    assert confidence >= 0.75, (
        f"Expected confidence>=0.75 for lofi/chill match, got {confidence:.2f} "
        f"(score {top_score:.2f})"
    )


# ---------------------------------------------------------------------------
# TEST GROUP 8 — Integration (hits Gemini API, optional)
# ---------------------------------------------------------------------------

def test_integration_full_pipeline():
    """Full pipeline: plain-English → recommendations. Checks structure only."""
    _ensure_index()
    result = run_agent("I want chill lofi music for studying", _FULL_SONGS, k=3)

    assert result.query, "AgentResult.query is empty"
    assert len(result.recommendations) == 3, (
        f"Expected 3 recs, got {len(result.recommendations)}"
    )
    assert len(result.steps) >= 4, (
        f"Expected >=4 observable steps, got {len(result.steps)}"
    )
    for rec in result.recommendations:
        assert 0.0 <= rec.confidence <= 1.0, f"Confidence {rec.confidence} out of range"
        assert rec.explanation, "Explanation is empty"
        assert rec.score > 0, "Score must be positive"

    # Top result should be low-energy (lofi/chill query)
    top = result.recommendations[0]
    assert top.song["energy"] <= 0.65, (
        f"Expected low-energy top result for chill query, got energy={top.song['energy']}"
    )


# ---------------------------------------------------------------------------
# pytest-compatible test functions (same logic, no harness scaffolding)
# ---------------------------------------------------------------------------

def test_score_perfect_match_pytest():     test_score_perfect_match()
def test_score_worst_match_pytest():       test_score_worst_match()
def test_score_bounded_pytest():           test_score_bounded()
def test_recommender_sorted_pytest():      test_recommender_sorted()
def test_guardrail_empty_pytest():         test_guardrail_empty_query()
def test_guardrail_short_pytest():         test_guardrail_short_query()
def test_fallback_valid_profile_pytest():  test_fallback_returns_valid_profile()
def test_fallback_genre_pytest():          test_fallback_genre_in_catalog()
def test_fallback_energy_pytest():         test_fallback_energy_range()


# ---------------------------------------------------------------------------
# Main — standalone harness runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FAST = "--fast" in sys.argv

    print("\n" + "=" * 65)
    print("  VIBEFINDER 2.0 — EVALUATION HARNESS")
    print("=" * 65)

    # Group 1: Scoring
    print("\n[Group 1] Scoring layer")
    _run("Score — perfect match ≥ 4.5",          test_score_perfect_match,        0.95)
    _run("Score — mismatch < 3.0",                test_score_worst_match,          0.95)
    _run("Score — always in [0, 5.0]",            test_score_bounded,              1.00)
    _run("Recommender — sorted by score",         test_recommender_sorted,         1.00)
    _run("Recommender — top result genre match",  test_recommender_top_genre_match, 0.95)

    # Group 2: RAG retriever
    print("\n[Group 2] RAG Retriever")
    _run("Retriever — returns k results",         test_retriever_returns_k,        1.00)
    _run("Retriever — similarity in [0, 1]",      test_retriever_similarity_range, 1.00)
    _run("Retriever — chill query → chill genres",test_retriever_chill_query,      0.85)
    _run("Retriever — intense query → high energy",test_retriever_intense_query,   0.85)
    _run("song_to_text — format check",           test_song_to_text_format,        1.00)

    # Group 3: Genre-lock
    print("\n[Group 3] Genre-lock detection & correction")
    _run("Genre-lock — detected on adversarial",  test_genre_lock_detected,        0.90)
    _run("Genre-lock — correction improves energy", test_genre_lock_correction_improves_energy, 0.90)
    _run("Genre-lock — no false positive",        test_no_genre_lock_for_fair_result, 0.90)

    # Group 4: Diversity
    print("\n[Group 4] Diversity")
    _run("Diversity — ≥ 2 genres in top-5",       test_diversity_at_least_two_genres, 0.90)
    _run("Diversity — preserves k",               test_diversity_preserves_k,      1.00)

    # Group 5: Guardrails
    print("\n[Group 5] Input guardrails")
    _run("Guardrail — empty query rejected",       test_guardrail_empty_query,      1.00)
    _run("Guardrail — short query rejected",       test_guardrail_short_query,      1.00)
    _run("Guardrail — whitespace-only rejected",   test_guardrail_whitespace_only,  1.00)

    # Group 6: Keyword fallback
    print("\n[Group 6] Keyword fallback (no API)")
    _run("Fallback — returns valid profile",       test_fallback_returns_valid_profile, 1.00)
    _run("Fallback — genre in catalog",            test_fallback_genre_in_catalog,  1.00)
    _run("Fallback — energy in [0, 1]",            test_fallback_energy_range,      1.00)
    _run("Fallback — detects acoustic keywords",   test_fallback_acoustic_detection, 0.90)
    _run("Fallback — high-energy keywords → ≥0.85",test_fallback_high_energy_keywords, 0.90)
    _run("Fallback — low-energy keywords → ≤0.45", test_fallback_low_energy_keywords, 0.90)

    # Group 7: Confidence scores
    print("\n[Group 7] Confidence scores")
    _run("Confidence — always in [0, 1]",         test_confidence_range,           1.00)
    _run("Confidence — ≥0.75 for strong match",   test_confidence_high_for_perfect_match, 0.85)

    # Group 8: Integration (optional)
    if not FAST:
        print("\n[Group 8] Integration — full pipeline (hits Gemini API)")
        _run("Integration — end-to-end pipeline",  test_integration_full_pipeline,  0.80)
    else:
        print("\n[Group 8] Integration — SKIPPED (--fast flag)")

    _print_summary()
