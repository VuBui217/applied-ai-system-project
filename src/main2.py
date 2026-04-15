"""
VibeFinder 2.0 — Interactive Entry Point

Run with:
    python -m src.main2
"""

import time
from src.recommender import load_songs
from src.retriever import build_index
from src.agent import run_agent, print_agent_result
from src.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    print("\n" + "=" * 60)
    print("  VibeFinder 2.0 — AI-Powered Music Discovery")
    print("=" * 60)

    # Load catalog and build/reuse ChromaDB index
    print("\nLoading catalog...")
    songs = load_songs("data/songs.csv")
    build_index(songs)
    print(f"Ready. {len(songs)} songs indexed.\n")
    print("Type your music request in plain English.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            query = input("What are you in the mood for? > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            result = run_agent(query, songs, k=5)
            print_agent_result(result)
        except ValueError as e:
            print(f"\n  Error: {e}\n")
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)
            print(f"\n  Something went wrong: {e}\n")


if __name__ == "__main__":
    main()
