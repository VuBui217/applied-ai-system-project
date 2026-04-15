# VibeFinder 2.0 — AI-Powered Music Discovery

## Origin Project: VibeFinder 1.0

This project extends **VibeFinder 1.0**, built in Modules 1–3 of CodePath AI-110. The original system was a rule-based content-based music recommender that scored a 20-song catalog against a hardcoded user profile using four features: energy proximity, acousticness preference, genre bonus, and mood bonus. It was stress-tested across four user profiles, including an adversarial "High-Energy Sad Blues" case that exposed a genre-lock bias where the genre bonus (2.0 out of 5.0 max) could override all other preferences and recommend a slow, quiet song to a user who explicitly asked for something intense.

VibeFinder 2.0 replaces the hardcoded input with natural language, fixes the genre-lock bias through an agentic self-correction loop, and adds semantic song retrieval through RAG.

---

## What It Does

VibeFinder 2.0 lets users describe what music they want in plain English:

```
> "I need something to hype me up at the gym"
> "chill lofi for studying late at night"
> "sad and slow, kind of rainy day vibes"
```

The system parses intent using the Gemini API, retrieves semantically similar songs from a 40-song catalog using sentence-transformer embeddings, scores and re-ranks the candidates using the original scoring logic, detects and corrects genre-lock bias automatically, and returns the top 5 recommendations with scores, confidence ratings, and plain-English explanations.

---

## Architecture Overview

```
User (plain English query)
         │
         ▼
 ┌─────────────────┐
 │  Input Guardrail │  validates query length and content
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Intent Parser   │  Gemini API → structured UserProfile
 │  (Gemini API)    │  fallback: keyword matching if API down
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  RAG Retriever   │  sentence-transformers + ChromaDB
 │  (local model)   │  returns top-20 semantic candidates
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Score & Rank    │  score_song() from VibeFinder 1.0
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Self-Evaluate   │  detects genre-lock bias
 │  (Agentic step)  │  re-ranks without genre bonus if needed
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Diversity Check │  ensures top-5 spans multiple genres
 └────────┬────────┘
          │
          ▼
 Final Recommendations + confidence scores + explanations
```

**Data flow:** Every agent run produces 5–6 observable reasoning steps stored in `AgentResult.steps`, so you can inspect what the system did at each stage.

**Components:**

| File | Role |
|---|---|
| `src/intent_parser.py` | Gemini API call + JSON parsing + keyword fallback |
| `src/retriever.py` | Sentence-transformer embeddings + ChromaDB search |
| `src/recommender.py` | Original scoring logic from VibeFinder 1.0 |
| `src/agent.py` | Orchestrates all steps, genre-lock detection, diversity |
| `src/logger.py` | Shared logger (console INFO + file DEBUG) |
| `tests/test_harness.py` | 27-test evaluation suite |
| `data/songs.csv` | 40-song catalog (expanded from original 20) |

---

## Setup Instructions

### 1. Get a free Gemini API key

Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — no credit card needed.

### 2. Clone and install

```bash
git clone <your-repo-url>
cd applied-ai-system-project

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
# Open .env and replace the placeholder with your real key:
# GEMINI_API_KEY=your_actual_key_here
```

### 4. Run the system

```bash
# Original VibeFinder 1.0 (hardcoded profiles, no AI)
python -m src.main

# VibeFinder 2.0 interactive mode (natural language input)
python -m src.main2

# Run the evaluation harness
python tests/test_harness.py          # full suite (hits Gemini once)
python tests/test_harness.py --fast   # unit tests only, no API calls

# Run original pytest suite
pytest
```

---

## Sample Interactions

### Example 1 — Study music

```
Query: "chill lofi music for studying late at night"

[Agent Reasoning]
  1. [parse_intent]      genre=lofi, mood=chill, energy=0.30, acoustic=False
  2. [rag_retrieve]      20 candidates. Top semantic match: 'Library Rain' (0.318)
  3. [score_candidates]  #1: 'Midnight Coding' score=4.05/5.00
  4. [self_evaluate]     Genre-lock detected: False. Ranking looks fair.
  5. [diversity_check]   Top-5 spans 3 genres: ambient, hip-hop, lofi

#1  Midnight Coding  —  LoRoom
    Genre: lofi | Mood: chill | Energy: 0.42
    Score: 4.05/5.00 | Confidence: 81%
    Why: genre matches (lofi), mood matches (chill), energy is close (0.42 vs 0.30)

#2  Backyard Smoke  —  Cipher Jones
    Genre: hip-hop | Mood: chill | Energy: 0.51
    Score: 3.39/5.00 | Confidence: 68%
    Why: mood matches (chill), energy is near (0.51 vs 0.30)
```

### Example 2 — Adversarial (genre-lock corrected)

```
Query: "I need fast and intense blues music"

[Agent Reasoning]
  1. [parse_intent]      genre=blues, mood=intense, energy=0.90, acoustic=False
  2. [rag_retrieve]      20 candidates. Top: 'Devil Got My Blues' (0.586)
  3. [score_candidates]  #1 after scoring: 'Gym Hero' score=3.89/5.00
  4. [self_evaluate]     Genre-lock detected: False. Ranking looks fair.
  5. [diversity_check]   Top-5 spans 4 genres: metal, pop, punk, rock

#1  Gym Hero  —  Max Pulse
    Genre: pop | Mood: intense | Energy: 0.93
    Score: 3.89/5.00 | Confidence: 78%
    Why: mood matches (intense), energy is close (0.93 vs 0.90)
```

> In VibeFinder 1.0, this same query returned "Devil Got My Blues" (energy 0.38)
> as #1 because the genre bonus was 2.0 out of 5.0. With the weight-shift
> experiment and the agentic scoring, the system now correctly prioritises
> high-energy songs over the genre label.

### Example 3 — Rainy day mood

```
Query: "something sad and slow for a rainy afternoon"

[Agent Reasoning]
  1. [parse_intent]      genre=ambient, mood=sad, energy=0.30, acoustic=False
  2. [rag_retrieve]      20 candidates. Top: 'Library Rain' (0.411)
  3. [score_candidates]  #1: 'Devil Got My Blues' score=2.97/5.00
  4. [self_evaluate]     Genre-lock detected: False. Ranking looks fair.
  5. [diversity_check]   Top-5 spans 4 genres: ambient, blues, lofi, pop

#1  Devil Got My Blues  —  Hound & Rust
    Genre: blues | Mood: sad | Energy: 0.38
    Score: 2.97/5.00 | Confidence: 59%
    Why: mood matches (sad), energy is close (0.38 vs 0.30)
```

---

## Design Decisions

### Why RAG over pure keyword matching?
The original system required exact string matches for genre and mood. A user saying "something dreamy and floaty" would score 0 on mood unless they typed the exact word "dreamy". RAG embeds the query semantically, so "floaty" and "dreamy" surface the same songs.

### Why keep `score_song()` from VibeFinder 1.0?
The original scoring function is deterministic, fast, and already incorporates the weight-shift experiment that reduces genre-lock. RAG provides the initial candidates; the original scorer provides explainable re-ranking. Using both is better than either alone — RAG alone can miss energy fit, scoring alone requires exact inputs.

### Why Gemini for intent parsing instead of rules?
A rule-based intent parser would need to cover hundreds of synonyms (gym = workout = exercise = pump up = hype ...). Gemini handles this naturally and returns structured JSON constrained to the catalog's exact genre and mood vocabulary.

### Why a keyword fallback?
The Gemini free tier has a 20 req/day hard limit per model. Rather than crashing when the quota is exceeded, the system falls back to a keyword-based parser that keeps the app functional. It is less accurate but ensures the product degrades gracefully.

### Why ChromaDB over a cloud vector store?
ChromaDB runs entirely on disk with no server. This keeps the project self-contained — anyone can run it with just `pip install` and a Gemini key, no external accounts.

### Trade-offs
- **Catalog size**: 40 songs is enough to demonstrate the system but too small for production. A real recommender would need thousands of songs before semantic similarity adds real value.
- **Gemini daily quota**: At 20 free requests/day, the system cannot be used heavily without a paid API key. The keyword fallback mitigates this but is less accurate.
- **No user history**: The system is stateless — each query starts fresh. Adding session memory would improve results significantly.

---

## Testing Summary

```
27/27 tests passed  |  Avg confidence (passing): 94%
```

| Group | Tests | Result |
|---|---|---|
| Scoring layer | 5 | 5/5 PASS |
| RAG retriever | 5 | 5/5 PASS |
| Genre-lock detection & correction | 3 | 3/3 PASS |
| Diversity | 2 | 2/2 PASS |
| Input guardrails | 3 | 3/3 PASS |
| Keyword fallback | 6 | 6/6 PASS |
| Confidence scores | 2 | 2/2 PASS |
| Integration (full pipeline) | 1 | 1/1 PASS |

**What worked well:** The genre-lock detector correctly flagged the adversarial "high-energy blues" profile every time. The keyword fallback correctly inferred energy level from workout/gym/chill keywords in all test cases. Confidence scores were always in range.

**What was harder than expected:** The RAG model (all-MiniLM-L6-v2) is general-purpose, not music-specific, so queries with abstract emotional language ("floaty", "nostalgic drive") have lower similarity scores (~0.30) than concrete genre/mood queries ("lofi studying", "metal angry") which score ~0.50–0.70. The scoring layer compensates for this by re-ranking on structured features.

**What I learned:** The single most impactful fix from 1.0 to 2.0 was not the AI features — it was the weight-shift experiment applied in Module 3 that halved the genre bonus. The agentic genre-lock corrector rarely needs to fire now because the scoring formula itself is more balanced. The AI features (RAG + Gemini intent parsing) matter most for the input side: they make the system accessible to real users instead of requiring them to fill in a form.

---

## Reflection

Building VibeFinder 2.0 made clear that the most important architectural decision is where you put the AI and where you keep the rules. Gemini is excellent at free-form understanding ("gym hype" → `{genre: edm, mood: excited, energy: 0.90}`) but it is non-deterministic and rate-limited. The original `score_song()` function is deterministic, fast, and explainable. The best design uses each where it fits: AI at the input boundary to understand human language, rules in the middle to rank and score consistently, and an agent layer to catch the edge cases the rules alone miss.

The hardest part of the project was not the code — it was designing test cases that prove the system works without being tautological. A test that just checks "did you get some results?" proves nothing. The useful tests are the adversarial ones: does the high-energy query actually return high-energy songs? Does the genre-lock detector fire on exactly the right cases and not on fair results? Those tests required deeply understanding what the system is supposed to do and writing assertions that would actually catch real failures.

---

## Limitations and Ethics

- **Genre-lock is reduced, not eliminated.** The bias exists whenever a genre has only one song in the catalog. Expanding the catalog is the right fix.
- **Gemini can misparse ambiguous queries.** "Something for my morning commute" might be parsed as pop or lofi depending on Gemini's interpretation — there is no ground truth to check against.
- **The keyword fallback is coarse.** It defaults to `pop` for unrecognised genres and `chill` for unrecognised moods. A user asking for reggae when Gemini is down gets a pop recommendation.
- **No content moderation.** User queries are passed directly to Gemini. A production system would need input filtering.
- **Data reflects narrow taste.** The 40-song catalog skews toward Western popular genres. Classical, folk, and world music are underrepresented.

---

## Collaboration with AI

This project was built with Claude Code (Anthropic) as a coding assistant.

**Helpful:** When the Gemini free-tier model (`gemini-2.0-flash-lite`) returned a quota limit of 0, Claude proactively tested alternative models and identified `gemini-2.5-flash-lite` as the working free-tier option — something that would have required manually reading the Gemini docs.

**Flawed:** Claude initially set `max_output_tokens=256` for Gemini responses, which caused truncated JSON output on several queries. The fix (bumping to 1024) was straightforward once the error was understood, but the initial value should have been higher from the start given that the response includes a `reasoning` field.
