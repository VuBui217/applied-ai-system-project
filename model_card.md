# Model Card — VibeFinder 2.0

---

## 1. Model Name

**VibeFinder 2.0** — AI-Powered Music Discovery System

Extends VibeFinder 1.0 (rule-based content-based recommender, Modules 1–3) with natural language input, semantic retrieval, and an agentic bias-correction loop.

---

## 2. Intended Use

VibeFinder 2.0 recommends songs from a 40-song catalog based on what a user describes in plain English. It is intended for classroom exploration of how real-world AI recommender systems work — combining language models, semantic search, and rule-based scoring. It assumes users have no prior knowledge of the catalog or music terminology. It is **not** a production recommender and is not designed for real users at scale.

---

## 3. How the System Works

Think of VibeFinder 2.0 as a pipeline with four distinct jobs:

1. **Understanding the user.** When a user types "something to hype me up at the gym," the system sends that to Google Gemini, a large language model. Gemini reads the request and returns a structured description: high energy (0.90), no acoustic preference, excited mood, edm genre. This step replaces the form fields required in VibeFinder 1.0.

2. **Finding candidates.** Every song in the 40-song catalog was pre-converted into a sentence — *"Gym Hero by Max Pulse — an intense pop track. Very high energy, fully electronic sound..."* — and embedded as a vector of numbers that captures its meaning. When a query arrives, the system embeds the query the same way and finds the 20 songs whose meaning is closest. This is called Retrieval-Augmented Generation (RAG).

3. **Scoring and ranking.** The 20 candidates are scored using the same formula from VibeFinder 1.0: energy proximity, acoustic preference, genre bonus (1.0 pts), and mood bonus (1.0 pts), maximum 5.0 points. The top results are kept.

4. **Self-checking.** The system inspects its own output. If the top-ranked song won mostly because of genre and mood bonuses but has a large energy mismatch, the system flags this as genre-lock bias, removes the genre bonus, and re-ranks. The final top 5 also go through a diversity check to ensure at least two different genres are represented.

---

## 4. Data

The catalog contains **40 songs** across 20 genres: pop, lofi, rock, metal, jazz, ambient, synthwave, indie pop, r&b, country, folk, edm, blues, reggae, hip-hop, classical, soul, punk, latin, trap.

Moods represented (18 total): happy, chill, intense, relaxed, romantic, angry, nostalgic, melancholic, euphoric, sad, laid-back, peaceful, confident, moody, focused, warm, dreamy, excited.

The catalog was expanded from 20 to 40 songs for VibeFinder 2.0. The additions focused on filling three gaps from 1.0:
- Mid-energy songs (0.45–0.70 range) — increased from 4 to 12
- Multi-song genres: lofi (5), pop (4), hip-hop (3), rock (3) now have enough variety for RAG to differentiate
- New genres: punk, latin, trap

**Whose taste does this catalog reflect?** The songs are fictional but the genres and attributes were chosen by the developer, reflecting a bias toward Western popular and electronic music. Classical, folk, world music, and non-English language genres are underrepresented.

---

## 5. Strengths

**Natural language input.** Users no longer need to know the system's vocabulary. "Rainy day vibes" maps correctly to low energy, sad mood — no form required.

**Genre-lock correction.** The adversarial "High-Energy Sad Blues" profile that caused VibeFinder 1.0 to recommend a slow, quiet song now triggers the genre-lock detector, which re-ranks to prioritise energy fit. This was the most significant failure mode identified in testing.

**Graceful degradation.** When the Gemini API is unavailable (quota exceeded, network error), the system falls back to keyword-based intent parsing and keeps running. The user still gets recommendations, just with less nuanced intent parsing.

**Explainable scoring.** Every recommendation includes a plain-English explanation: *"Score 4.05/5.00 — genre matches (lofi), mood matches (chill), energy is close (0.42 vs 0.30)."*

**Observable agent reasoning.** Each run logs 5–6 intermediate steps, making it possible to understand exactly why each recommendation was made.

---

## 6. Limitations and Bias

**Catalog sparsity still limits quality.** 40 songs is an improvement over 20, but 9 genres still have only one song. For those genres, the top result is nearly predetermined regardless of energy or mood fit.

**Genre-lock is reduced, not eliminated.** The genre bonus (1.0/5.0 = 20% of max score) can still outweigh energy fit for users whose target energy is moderate (0.4–0.7). The detector only flags cases where the energy deviation exceeds 0.30 and the bonus contribution is large — smaller mismatches pass through uncorrected.

**RAG model is not music-specific.** The sentence-transformer model (`all-MiniLM-L6-v2`) was trained on general English text. It handles concrete musical descriptions ("jazz relaxed low energy acoustic") well but struggles with abstract emotional language ("something that sounds like driving home alone at 2am"). Similarity scores for abstract queries are lower (~0.25–0.35) compared to concrete queries (~0.50–0.70).

**Binary acoustic preference.** The `likes_acoustic` field is still True or False. Users who enjoy both acoustic and electronic music are forced into one camp.

**Gemini daily quota.** The free tier allows 20 requests/day per model. Heavy testing depletes this quickly. The keyword fallback defaults to `pop` for unrecognised genres, which may produce irrelevant results.

**No personalisation.** The system has no memory of past queries or feedback. Each session starts fresh. In a real recommender, user history is one of the strongest signals.

---

## 7. Evaluation

### Automated test harness

```
27/27 tests passed  |  Avg confidence (passing): 94%
```

Tests cover: scoring correctness, RAG retrieval accuracy, genre-lock detection and correction, diversity, input guardrails, keyword fallback, confidence score bounds, and end-to-end integration.

### Profile-based evaluation (manual)

| Profile | Query | Top Result | Energy Match | Notes |
|---|---|---|---|---|
| Study | "chill lofi for studying" | Midnight Coding (lofi/chill, 0.42) | Good (target 0.30) | Correct |
| Workout | "high energy morning run" | Neon Carnival (pop/excited, 0.88) | Good (target 0.85) | Correct |
| Rainy day | "sad and slow rainy afternoon" | Devil Got My Blues (blues/sad, 0.38) | Good (target 0.30) | Correct |
| Adversarial | "fast intense blues music" | Gym Hero (pop/intense, 0.93) | Good (target 0.90) | Genre-lock corrected ✓ |

**Key finding:** The adversarial profile that broke VibeFinder 1.0 (recommending energy 0.38 to a user who wanted 0.90) now works correctly. The combination of the weight-shift experiment (from Module 3) and the agentic genre-lock detector ensures the high-energy result wins.

### Confidence score distribution

| Score bracket | Meaning | Typical cases |
|---|---|---|
| ≥ 80% | Strong match | Genre + mood + energy all align |
| 60–79% | Good match | 2 of 3 main features align |
| 40–59% | Partial match | 1 feature matches or near-miss on multiple |
| < 40% | Weak match | Shown only when catalog lacks better options |

### What surprised me

The keyword fallback proved more robust than expected. Because the catalog moods and genres closely track common English vocabulary, simple substring matching handles ~70% of real queries correctly. The remaining ~30% benefit from Gemini's ability to understand context (e.g. "workout" → high energy, "commute" → moderate energy, "pre-game" → high energy).

---

## 8. Future Work

- **Expand the catalog.** 400+ songs would give RAG real differentiation power within each genre.
- **Use a music-specific embedding model.** A model trained on music descriptions (e.g. fine-tuned on MusicBrainz tags or Spotify metadata) would produce better similarity scores for abstract emotional queries.
- **Add session memory.** "More like that last one" or "less pop, more jazz" would allow iterative refinement — the key missing piece for a real product.
- **Use the ignored features.** `valence`, `danceability`, and `tempo_bpm` are stored in the catalog but not used in scoring. A user asking for "something to dance to" has no way to express danceability preference today.
- **Fuzzy mood matching.** "Chill" and "laid-back" are semantically identical but score 0 vs 1.0 for mood. A soft similarity score (e.g. cosine distance between mood embeddings) would reduce this all-or-nothing behaviour.
- **Multi-turn conversation.** Allow users to refine results: "less electronic," "similar energy but happier."

---

## 9. Personal Reflection

The biggest lesson from VibeFinder 2.0 is that a good AI system is not the one with the most AI in it — it is the one where each component does what it is best at. Gemini is better at understanding language than any rule system I could write. A sentence-transformer is better at finding semantically similar descriptions than keyword matching. But the original `score_song()` function is better than either of them at the specific task of ranking songs by numerical feature proximity, because it was designed exactly for that job and its behavior is transparent and predictable.

The second lesson is about failure modes. The genre-lock bias in VibeFinder 1.0 was not a flaw in the code — the code did exactly what it was told to do. The flaw was in the implicit assumption that genre was twice as important as energy. When you build scoring systems, the weights you choose are a statement about values, and those statements can be wrong. Building the agentic self-evaluator forced me to make that assumption explicit and gave me a mechanism to catch it when it produces a bad result.

---

## Collaboration with AI

This project was developed with Claude Code (Anthropic) as a coding assistant.

**Instance where AI was helpful:** When the initially chosen Gemini model (`gemini-2.0-flash-lite`) returned a quota of 0, Claude automatically tested three alternative models and identified `gemini-2.5-flash-lite` as the working free-tier option — saving significant debugging time.

**Instance where AI was flawed:** Claude initially set `max_output_tokens=256` in the Gemini API call. Because the response includes a `reasoning` field with a full sentence, the JSON was regularly truncated mid-object, causing parse failures. The correct value was 1024. The fix was simple once identified, but the initial configuration should have been validated against a real response before committing to it.
