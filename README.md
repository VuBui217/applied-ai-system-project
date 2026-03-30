# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

VibeFinder 1.0 scores every song in a 20-song catalog against a user's stated preferences — favorite genre, mood, target energy, and acoustic preference — and returns the top 5 matches. The system was stress-tested against four profiles including an adversarial "High-Energy Sad Blues" edge case that revealed a genre-lock bias in the scoring logic. A weight-shift experiment (doubled energy, halved genre bonus) was applied to study how sensitive the rankings are to each feature's relative importance.

---

## How The System Works

Real-world recommenders like Spotify combine collaborative filtering (learning from millions of users with similar taste) and content-based filtering (analyzing the actual properties of each song). My version focuses on content-based filtering: it compares the attributes of each song directly against what the user says they prefer, without relying on any other users' behavior. For each song, the system computes a proximity score — the closer a song's attributes are to the user's preferences, the higher it scores. After scoring every song in the catalog, the system ranks them and returns the top matches. This approach is transparent and easy to reason about, which makes it a good starting point before adding more complex signals.

### Song Features

Each `Song` object stores the following attributes:

- `energy` — how intense or active the track feels (0.0 to 1.0)
- `acousticness` — how acoustic vs. electronic the song is (0.0 to 1.0)
- `valence` — emotional positivity; high = uplifting, low = melancholic (0.0 to 1.0)
- `danceability` — how suitable the track is for dancing (0.0 to 1.0)
- `genre` — broad style category (pop, lofi, rock, ambient, jazz, synthwave, indie pop, etc.)
- `mood` — descriptive feel of the track (happy, chill, intense, relaxed, moody, focused, etc.)
- `tempo_bpm` — beats per minute (stored but not weighted heavily at this scale)

### UserProfile Features

Each `UserProfile` stores:

- `target_energy` — the user's preferred energy level (0.0 to 1.0)
- `likes_acoustic` — whether the user prefers acoustic over electronic sounds (True or False). True favors songs with high acousticness; False favors songs with low acousticness.
- `favorite_genre` — the genre the user most wants to hear
- `favorite_mood` — the mood the user is currently in

### Scoring Rule (per song)

Each numeric feature gets a proximity score: `1 - |user_preference - song_value|`, which ranges from 0.0 to 1.0. Categorical matches add a fixed bonus on top.

```
energy_score   = 1 - |target_energy - song.energy|
acoustic_score = song.acousticness       if likes_acoustic is True
               = 1 - song.acousticness   if likes_acoustic is False

genre_bonus    = +2.0  if song.genre == favorite_genre, else 0
mood_bonus     = +1.0  if song.mood  == favorite_mood,  else 0

total_score    = energy_score + acoustic_score + genre_bonus + mood_bonus
```

Maximum possible score is **5.0** (1.0 energy + 1.0 acoustic + 2.0 genre + 1.0 mood). Genre outweighs mood 2:1 because genre is a harder filter — a metal fan rarely enjoys jazz regardless of energy fit. Mood is a softer signal that complements the numeric scores.

### Ranking Rule (across all songs)

All songs are scored, then sorted by `total_score` descending. The top `k` songs (default 5) are returned as recommendations.

### Potential Biases in This Scoring Rule

- **Genre dominance**: The genre bonus (+2.0) is 40% of the maximum possible score (5.0). A song in the right genre but with a poor energy and acoustic fit can still outscore a song that matches the user's numeric preferences perfectly in a different genre. Users with niche genre preferences may always get the same 1–2 songs at the top regardless of how well those songs actually fit their vibe.

- **Catalog sparsity amplifies genre lock-in**: Most genres in this catalog appear only once across 20 songs. That single genre-matching song gets a +2.0 head start no matter how well it actually fits the user, making it very hard for other songs to compete.

- **Binary acoustic preference**: `likes_acoustic` is True or False with no middle ground. Users who enjoy both acoustic and electronic sounds are forced to pick a side, which systematically disadvantages songs on the opposite end even if the user would genuinely enjoy them.

- **Exact mood matching**: Mood comparisons are all-or-nothing. Similar moods like "chill" and "laid-back", or "happy" and "euphoric", are treated as completely different. A song that is a near-perfect mood match scores 0 for mood while an exact string match scores +1.0.

- **Valence and danceability are invisible**: Songs store `valence` and `danceability` but the `UserProfile` has no fields to express preferences for those dimensions. The scoring formula ignores them entirely, so a high-energy dance track and a high-energy ballad score identically if their genre, mood, and acousticness match.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

   ```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Full terminal output for all runs is captured in [results.md](results.md).

### Profiles tested

Four user profiles were run against the recommender:

| Profile | genre | mood | energy | acoustic |
|---|---|---|---|---|
| High-Energy Pop | pop | happy | 0.9 | False |
| Chill Lofi | lofi | chill | 0.35 | True |
| Deep Intense Rock | rock | intense | 0.9 | False |
| Adversarial: High-Energy Sad Blues | blues | sad | 0.95 | False |

The adversarial profile was designed to expose contradictory preferences — a user who wants very high-energy music but in a typically slow/quiet genre (blues) with a sad mood.

### Weight-shift experiment (Step 3)

Changed `recommender.py` scoring:
- **Energy weight doubled**: `energy_score = 2 * (1 - |target - song.energy|)` (max 2.0 instead of 1.0)
- **Genre bonus halved**: `genre_bonus = 1.0` if match (was 2.0)
- Total max score stays at 5.0 — math remains valid

**Result:** For the adversarial profile, the winning margin for "Devil Got My Blues" shrank from **1.59 points → 0.02 points** over the next best song. High-energy songs nearly overtook it, confirming the original genre bonus was masking energy mismatches. For the Chill Lofi profile, "Spacewalk Thoughts" rose from #4 to #3 because its mood match became proportionally more valuable relative to the smaller genre bonus.

---

## Limitations and Risks

- **Genre lock-in**: The genre bonus (originally 2.0 out of 5.0 max) is so large it can override every other preference. The adversarial test proved this: a user asking for energy 0.95 received a song with energy 0.38 simply because it matched genre and mood.
- **Catalog too small**: 20 songs means most genres have exactly one representative. Once the genre bonus is awarded, the winner is nearly predetermined.
- **Missing features**: `valence`, `danceability`, and `tempo_bpm` are stored in the CSV but never used in scoring. A dance track and a ballad with the same energy level score identically.
- **Binary acoustic preference**: No middle ground — users who enjoy both acoustic and electronic sounds are forced to pick a side.
- **Exact mood matching only**: "chill" and "laid-back" are treated as completely different, even though a listener would likely enjoy both.

See [model_card.md](model_card.md) for a deeper analysis.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this

---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:

- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:

- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:

- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"
```
