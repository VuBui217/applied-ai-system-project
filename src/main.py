"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


# --- User Profiles ---

# Profile 1: High-Energy Pop fan who loves upbeat, danceable hits
HIGH_ENERGY_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.9,
    "likes_acoustic": False,
}

# Profile 2: Chill Lofi listener who studies or relaxes with soft, acoustic sounds
CHILL_LOFI = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.35,
    "likes_acoustic": True,
}

# Profile 3: Deep Intense Rock — heavy riffs, high energy, aggressive mood
DEEP_INTENSE_ROCK = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.9,
    "likes_acoustic": False,
}

# Profile 4 (Edge Case / Adversarial): Conflicting — very high energy but wants a sad mood.
# This tests whether the system can reconcile opposite preferences.
CONFLICTING_HIGH_ENERGY_SAD = {
    "favorite_genre": "blues",
    "favorite_mood": "sad",
    "target_energy": 0.95,
    "likes_acoustic": False,
}

PROFILES = [
    ("High-Energy Pop",            HIGH_ENERGY_POP),
    ("Chill Lofi",                 CHILL_LOFI),
    ("Deep Intense Rock",          DEEP_INTENSE_ROCK),
    ("Adversarial: High-Energy Sad Blues", CONFLICTING_HIGH_ENERGY_SAD),
]


def print_recommendations(label: str, recommendations) -> None:
    print("\n" + "=" * 55)
    print(f"  PROFILE: {label}")
    print("=" * 55)
    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{i}  {song['title']}  —  {song['artist']}")
        print(f"    Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}")
        print(f"    Score: {score:.2f} / 5.00")
        reason_text = explanation.split(": ", 1)[-1] if ": " in explanation else explanation
        print(f"    Why:   {reason_text}")
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")

    for label, user_prefs in PROFILES:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(label, recommendations)

    print("=" * 55)


if __name__ == "__main__":
    main()
