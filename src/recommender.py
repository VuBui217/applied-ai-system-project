import csv
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_dict = asdict(user)
        scored = sorted(self.songs, key=lambda s: score_song(user_dict, asdict(s)), reverse=True)
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_dict = asdict(user)
        song_dict = asdict(song)
        total = score_song(user_dict, song_dict)
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre matches ({song.genre})")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood matches ({song.mood})")
        energy_diff = abs(user.target_energy - song.energy)
        if energy_diff <= 0.15:
            reasons.append(f"energy is close ({song.energy:.2f} vs target {user.target_energy:.2f})")
        if not reasons:
            reasons.append("closest overall match available")
        return f"Score {total:.2f}: " + ", ".join(reasons) + "."

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, csv_path)
    songs = []
    with open(full_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    print(f"Loaded songs: {len(songs)}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> float:
    """
    Scores a single song against a user's preferences.
    Returns a float between 0.0 and 5.0.
    """
    # EXPERIMENT (Step 3): doubled energy weight, halved genre weight.
    # Original: energy×1, genre bonus 2.0. New: energy×2, genre bonus 1.0.
    # Total max remains 5.0 (2.0 + 1.0 + 1.0 + 1.0).
    energy_score = 2 * (1 - abs(user_prefs["target_energy"] - song["energy"]))

    if user_prefs["likes_acoustic"]:
        acoustic_score = song["acousticness"]
    else:
        acoustic_score = 1 - song["acousticness"]

    genre_bonus = 1.0 if song["genre"] == user_prefs["favorite_genre"] else 0.0
    mood_bonus  = 1.0 if song["mood"]  == user_prefs["favorite_mood"]  else 0.0

    return energy_score + acoustic_score + genre_bonus + mood_bonus


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = []
    for song in songs:
        score = score_song(user_prefs, song)

        reasons = []
        if song["genre"] == user_prefs["favorite_genre"]:
            reasons.append(f"genre matches ({song['genre']})")
        if song["mood"] == user_prefs["favorite_mood"]:
            reasons.append(f"mood matches ({song['mood']})")
        if abs(user_prefs["target_energy"] - song["energy"]) <= 0.15:
            reasons.append(f"energy is close ({song['energy']:.2f} vs target {user_prefs['target_energy']:.2f})")
        if not reasons:
            reasons.append("closest overall match available")
        explanation = f"Score {score:.2f}: " + ", ".join(reasons) + "."

        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
