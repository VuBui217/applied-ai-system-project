# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

VibeFinder 1.0 recommends songs from a 20-song catalog based on a listener's stated preferences. It is intended for classroom exploration of how rule-based scoring systems work in music recommendation. It assumes users can clearly describe what they want — a favorite genre, a target mood, an energy level, and whether they like acoustic sounds. It is not a production recommender; it is a teaching tool.

---

## 3. How the Model Works

Think of the model like a helpful friend who gives each song a score out of 5. The score is built from four ingredients:

1. **Energy match** — How close is the song's energy level to what you asked for? A perfect match adds up to 1 point (or 2 points after the weight-shift experiment).
2. **Acoustic preference** — Do you like acoustic sounds? The model rewards songs that fit your preference, up to 1 point.
3. **Genre bonus** — If the song's genre matches your favorite, it gets a 2-point bonus (1 point after the experiment). This is the single biggest driver of the score.
4. **Mood bonus** — If the song's mood matches your favorite, it gets 1 extra point.

The model adds all four parts together and returns the five songs with the highest totals.

---

## 4. Data

The catalog contains 20 songs spanning 14 genres: pop, lofi, rock, metal, jazz, ambient, synthwave, indie pop, r&b, country, folk, edm, blues, reggae, hip-hop, classical, soul. Moods represented include happy, chill, intense, focused, romantic, angry, nostalgic, melancholic, euphoric, sad, laid-back, peaceful, confident, moody, relaxed, and warm.

Pop and lofi each have 2–3 songs, while most other genres have exactly one. No songs were added or removed from the starter dataset. The dataset is too small to provide variety for niche genres, and certain emotional states (e.g., anxious, bittersweet, excited) have no representation at all.

---

## 5. Strengths

The system works best when the user's preferences align tightly with an existing song in the catalog. The "Chill Lofi" profile produced very intuitive results — the top two picks were both lofi tracks with matching genre, mood, and near-perfect energy levels. The "Deep Intense Rock" profile also performed well, ranking "Storm Runner" first with a near-perfect score of 4.89/5. For users whose genre has multiple songs in the catalog (pop, lofi), the top results feel genuinely differentiated and reasonable.

---

## 6. Limitations and Bias

The system has a strong genre-lock bias: the genre bonus (originally 2.0 points out of a max 5.0) is so large that it can override every other preference. The most revealing example came from the adversarial "High-Energy Sad Blues" profile: a user who said they wanted high-energy music (target: 0.95) received "Devil Got My Blues" as their top pick — a slow blues track with an energy of only 0.38. The song won because its genre and mood matched perfectly, earning 3.0 points, while high-energy songs without genre/mood matches could only earn a maximum of ~2.0 points. In plain terms: the model will confidently recommend a slow, quiet song to someone who just asked for something intense and loud, as long as the genre label lines up.

A second weakness is catalog imbalance. Over 60% of genres have only one representative song, so once the genre bonus is awarded, the winner is almost always predetermined. Finally, the model ignores important musical dimensions like tempo, danceability, and valence entirely — features that are present in the dataset but not used in the score.

---

## 7. Evaluation

**Profiles tested:**

| Profile | Key Preferences | Top Result |
|---|---|---|
| High-Energy Pop | genre=pop, mood=happy, energy=0.9 | Sunrise City (pop/happy/0.82) |
| Chill Lofi | genre=lofi, mood=chill, energy=0.35 | Library Rain (lofi/chill/0.35) |
| Deep Intense Rock | genre=rock, mood=intense, energy=0.9 | Storm Runner (rock/intense/0.91) |
| Adversarial: High-Energy Sad Blues | genre=blues, mood=sad, energy=0.95 | Devil Got My Blues (blues/sad/0.38) |

**What surprised me:**

The adversarial profile exposed the genre-lock problem in the starkest way. A user asking for high-energy music landed "Devil Got My Blues" (energy 0.38 out of 1.0) as their #1 recommendation. The reason is simple math: genre + mood = 3.0 points, which beats any song that only matches on energy. From a listener's perspective, this is a real failure — someone at the gym who wanted "sad blues vibes but fast" would get the slowest, quietest song in the catalog.

**Weight-shift experiment (Step 3):**

I doubled the energy weight and halved the genre bonus. Under the new math (energy×2, genre bonus 1.0), the total max stays at 5.0. The adversarial profile's top pick stayed the same ("Devil Got My Blues"), but its winning margin collapsed from a comfortable 1.59 points down to just 0.02 points above "Strobe Garden." A few more high-energy blues songs in the dataset would almost certainly have flipped the result. For the "Chill Lofi" profile, the reordering brought "Spacewalk Thoughts" (ambient/chill) up from #4 to #3, rewarding its mood match more visibly. The experiment confirmed that energy sensitivity increases with the doubled weight, making recommendations more precise for energy-focused users while slightly reducing the genre lock.

**Why "Gym Hero" keeps showing up:**

"Gym Hero" (pop, intense, energy 0.93) appears in the top 5 for both the High-Energy Pop and Deep Intense Rock profiles. It ranks second for High-Energy Pop because it shares the genre bonus but misses on mood. It ranks second for Deep Intense Rock because it has the right mood and the right energy — just the wrong genre. In plain language: "Gym Hero" is a high-energy song that sits at the crossroads between what both profiles want. The scoring system treats it as a reasonable runner-up in both cases, which actually feels correct.

---

## 8. Future Work

- Use the tempo_bpm, valence, and danceability columns that already exist in the dataset but are currently ignored.
- Expand the catalog — 20 songs is too small; single-song genres make the genre bonus a near-certainty for some profiles.
- Add a diversity penalty so the top 5 results don't cluster in the same genre.
- Allow users to specify importance weights themselves (e.g., "energy matters more to me than genre today").
- Handle contradictory preferences gracefully rather than letting the highest-point bonus win unconditionally.

---

## 9. Personal Reflection

Building VibeFinder 1.0 made clear how quickly a simple scoring rule can produce confidently wrong answers. The genre bonus felt reasonable in isolation — genre is a strong signal for music taste — but giving it twice the weight of every other feature turns it into an unchecked override. The adversarial profile was the most educational moment: watching the system recommend a slow blues track to someone who explicitly wanted high energy showed exactly how a recommender can appear to be working (it matched two fields perfectly) while completely failing the user's actual intent. Real music apps solve this with collaborative filtering and learned embeddings, but even a rule-based toy system teaches the core lesson: the features you choose and the weights you give them shape not just the output but whose preferences the system actually serves.
