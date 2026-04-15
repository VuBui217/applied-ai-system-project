"""
RAG Retriever for VibeFinder 2.0

Converts each song into a natural-language description, embeds them with
sentence-transformers (all-MiniLM-L6-v2, runs locally for free), and stores
the vectors in a persistent ChromaDB collection.

Public API
----------
build_index(songs)          -> None          (call once to populate ChromaDB)
retrieve(query, k)          -> List[Dict]    (semantic search, returns song dicts)
"""

import os
import csv
import warnings
from typing import List, Dict

# Suppress noisy HuggingFace / tokenizer startup messages
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)

import chromadb
from chromadb.config import Settings
import transformers
transformers.logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_DIR = os.path.join(_BASE_DIR, "data", "chroma_db")
_COLLECTION_NAME = "vibefinder_songs"

# ---------------------------------------------------------------------------
# Model (loaded once at import time so it is not reloaded on every call)
# ---------------------------------------------------------------------------
_model = SentenceTransformer("all-MiniLM-L6-v2")


def _energy_label(e: float) -> str:
    if e < 0.35:
        return "very low energy"
    if e < 0.55:
        return "low to moderate energy"
    if e < 0.72:
        return "moderate energy"
    if e < 0.88:
        return "high energy"
    return "very high energy"


def _acoustic_label(a: float) -> str:
    if a >= 0.75:
        return "very acoustic and organic"
    if a >= 0.50:
        return "somewhat acoustic"
    if a >= 0.25:
        return "a mix of acoustic and electronic"
    return "fully electronic"


def _valence_label(v: float) -> str:
    if v >= 0.75:
        return "very positive and uplifting"
    if v >= 0.50:
        return "generally positive"
    if v >= 0.30:
        return "somewhat dark or bittersweet"
    return "dark and heavy"


def _dance_label(d: float) -> str:
    if d >= 0.80:
        return "highly danceable"
    if d >= 0.60:
        return "moderately danceable"
    return "not very danceable"


def song_to_text(song: Dict) -> str:
    """
    Convert a song dict into a descriptive sentence that embeds well.

    Example output:
      "Storm Runner by Voltline — a intense rock track. Very high energy,
       fully electronic sound. Dark and heavy feel. Moderately danceable.
       Tempo: 152 BPM."
    """
    return (
        f"{song['title']} by {song['artist']} — "
        f"a {song['mood']} {song['genre']} track. "
        f"{_energy_label(float(song['energy'])).capitalize()}, "
        f"{_acoustic_label(float(song['acousticness']))} sound. "
        f"{_valence_label(float(song['valence'])).capitalize()} feel. "
        f"{_dance_label(float(song['danceability'])).capitalize()}. "
        f"Tempo: {song['tempo_bpm']} BPM."
    )


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_client() -> chromadb.PersistentClient:
    os.makedirs(_CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=_CHROMA_DIR)


def build_index(songs: List[Dict]) -> None:
    """
    Embed all songs and upsert them into the ChromaDB collection.
    Safe to call multiple times — existing entries are overwritten, not duplicated.
    """
    client = _get_client()
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)

    texts = [song_to_text(s) for s in songs]
    embeddings = _model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[str(s["id"]) for s in songs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "id":           int(s["id"]),
                "title":        s["title"],
                "artist":       s["artist"],
                "genre":        s["genre"],
                "mood":         s["mood"],
                "energy":       float(s["energy"]),
                "tempo_bpm":    float(s["tempo_bpm"]),
                "valence":      float(s["valence"]),
                "danceability": float(s["danceability"]),
                "acousticness": float(s["acousticness"]),
            }
            for s in songs
        ],
    )


def retrieve(query: str, k: int = 10) -> List[Dict]:
    """
    Embed the query and return the top-k most semantically similar songs.

    Parameters
    ----------
    query : str
        Plain-English description, e.g. "something chill for late-night studying"
    k : int
        Number of results to return (default 10 so the agent has room to re-rank)

    Returns
    -------
    List of song metadata dicts, ordered by semantic similarity (closest first).
    """
    client = _get_client()
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Call build_index(songs) first."
        )

    query_embedding = _model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    songs = []
    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        song = dict(meta)
        # Convert cosine distance (0=identical, 2=opposite) to similarity (0–1)
        song["similarity"] = round(1 - dist / 2, 4)
        song["description"] = doc
        songs.append(song)

    return songs


# ---------------------------------------------------------------------------
# CLI smoke-test  (python -m src.retriever)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.recommender import load_songs

    print("Loading songs and building index...")
    raw_songs = load_songs("data/songs.csv")
    build_index(raw_songs)
    print(f"Indexed {len(raw_songs)} songs.\n")

    test_queries = [
        "something chill and relaxing for studying late at night",
        "high energy workout music, really intense",
        "sad and slow music for a rainy day",
        "upbeat happy pop song for a road trip",
        "dark and angry heavy metal",
    ]

    for q in test_queries:
        print(f"Query: \"{q}\"")
        hits = retrieve(q, k=3)
        for i, s in enumerate(hits, 1):
            print(f"  #{i} {s['title']} ({s['genre']}/{s['mood']}) "
                  f"energy={s['energy']:.2f}  similarity={s['similarity']:.3f}")
        print()
