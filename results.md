# Evaluation Results — Terminal Output

All runs executed from the `src/` directory with the virtual environment active.

---

## Baseline Run (original single starter profile)

```
(.venv) tacmeics@Vus-MacBook-Pro src % python main.py
Loaded songs: 20

==================================================
  TOP RECOMMENDATIONS
==================================================

#1  Storm Runner  —  Voltline
    Genre: rock  |  Mood: intense  |  Energy: 0.91
    Score: 4.79 / 5.00
    Why:   genre matches (rock), mood matches (intense), energy is close (0.91 vs target 0.80).

#2  Gym Hero  —  Max Pulse
    Genre: pop  |  Mood: intense  |  Energy: 0.93
    Score: 2.82 / 5.00
    Why:   mood matches (intense), energy is close (0.93 vs target 0.80).

#3  Block by Block  —  Cipher Jones
    Genre: hip-hop  |  Mood: confident  |  Energy: 0.78
    Score: 1.86 / 5.00
    Why:   energy is close (0.78 vs target 0.80).

#4  Strobe Garden  —  Ultrawave
    Genre: edm  |  Mood: euphoric  |  Energy: 0.95
    Score: 1.82 / 5.00
    Why:   energy is close (0.95 vs target 0.80).

#5  Sunrise City  —  Neon Echo
    Genre: pop  |  Mood: happy  |  Energy: 0.82
    Score: 1.80 / 5.00
    Why:   energy is close (0.82 vs target 0.80).

==================================================
```

**Observation:** Storm Runner dominates with a near-perfect score of 4.79 because it matches genre, mood, and energy simultaneously. The #2–5 songs all score below 2.82 — a gap of almost 2 full points — showing how decisive the genre bonus is when only one rock song exists in the catalog.

---

## Step 3 Weight-Shift Experiment — All Four Profiles

Settings: `energy × 2`, `genre bonus 1.0` (halved). Total max remains 5.0.

```
(.venv) tacmeics@Vus-MacBook-Pro src % python main.py
Loaded songs: 20

=======================================================
  PROFILE: High-Energy Pop
=======================================================

#1  Sunrise City  —  Neon Echo
    Genre: pop  |  Mood: happy  |  Energy: 0.82
    Score: 4.66 / 5.00
    Why:   genre matches (pop), mood matches (happy), energy is close (0.82 vs target 0.90).

#2  Gym Hero  —  Max Pulse
    Genre: pop  |  Mood: intense  |  Energy: 0.93
    Score: 3.89 / 5.00
    Why:   genre matches (pop), energy is close (0.93 vs target 0.90).

#3  Rooftop Lights  —  Indigo Parade
    Genre: indie pop  |  Mood: happy  |  Energy: 0.76
    Score: 3.37 / 5.00
    Why:   mood matches (happy), energy is close (0.76 vs target 0.90).

#4  Storm Runner  —  Voltline
    Genre: rock  |  Mood: intense  |  Energy: 0.91
    Score: 2.88 / 5.00
    Why:   energy is close (0.91 vs target 0.90).

#5  Strobe Garden  —  Ultrawave
    Genre: edm  |  Mood: euphoric  |  Energy: 0.95
    Score: 2.87 / 5.00
    Why:   energy is close (0.95 vs target 0.90).


=======================================================
  PROFILE: Chill Lofi
=======================================================

#1  Library Rain  —  Paper Lanterns
    Genre: lofi  |  Mood: chill  |  Energy: 0.35
    Score: 4.86 / 5.00
    Why:   genre matches (lofi), mood matches (chill), energy is close (0.35 vs target 0.35).

#2  Midnight Coding  —  LoRoom
    Genre: lofi  |  Mood: chill  |  Energy: 0.42
    Score: 4.57 / 5.00
    Why:   genre matches (lofi), mood matches (chill), energy is close (0.42 vs target 0.35).

#3  Spacewalk Thoughts  —  Orbit Bloom
    Genre: ambient  |  Mood: chill  |  Energy: 0.28
    Score: 3.78 / 5.00
    Why:   mood matches (chill), energy is close (0.28 vs target 0.35).

#4  Focus Flow  —  LoRoom
    Genre: lofi  |  Mood: focused  |  Energy: 0.40
    Score: 3.68 / 5.00
    Why:   genre matches (lofi), energy is close (0.40 vs target 0.35).

#5  Shallow River  —  The Maren
    Genre: folk  |  Mood: melancholic  |  Energy: 0.31
    Score: 2.86 / 5.00
    Why:   energy is close (0.31 vs target 0.35).


=======================================================
  PROFILE: Deep Intense Rock
=======================================================

#1  Storm Runner  —  Voltline
    Genre: rock  |  Mood: intense  |  Energy: 0.91
    Score: 4.88 / 5.00
    Why:   genre matches (rock), mood matches (intense), energy is close (0.91 vs target 0.90).

#2  Gym Hero  —  Max Pulse
    Genre: pop  |  Mood: intense  |  Energy: 0.93
    Score: 3.89 / 5.00
    Why:   mood matches (intense), energy is close (0.93 vs target 0.90).

#3  Strobe Garden  —  Ultrawave
    Genre: edm  |  Mood: euphoric  |  Energy: 0.95
    Score: 2.87 / 5.00
    Why:   energy is close (0.95 vs target 0.90).

#4  Iron Cathedral  —  Dreadwall
    Genre: metal  |  Mood: angry  |  Energy: 0.97
    Score: 2.82 / 5.00
    Why:   energy is close (0.97 vs target 0.90).

#5  Sunrise City  —  Neon Echo
    Genre: pop  |  Mood: happy  |  Energy: 0.82
    Score: 2.66 / 5.00
    Why:   energy is close (0.82 vs target 0.90).


=======================================================
  PROFILE: Adversarial: High-Energy Sad Blues
=======================================================

#1  Devil Got My Blues  —  Hound & Rust
    Genre: blues  |  Mood: sad  |  Energy: 0.38
    Score: 2.99 / 5.00
    Why:   genre matches (blues), mood matches (sad).

#2  Strobe Garden  —  Ultrawave
    Genre: edm  |  Mood: euphoric  |  Energy: 0.95
    Score: 2.97 / 5.00
    Why:   energy is close (0.95 vs target 0.95).

#3  Iron Cathedral  —  Dreadwall
    Genre: metal  |  Mood: angry  |  Energy: 0.97
    Score: 2.92 / 5.00
    Why:   energy is close (0.97 vs target 0.95).

#4  Gym Hero  —  Max Pulse
    Genre: pop  |  Mood: intense  |  Energy: 0.93
    Score: 2.91 / 5.00
    Why:   energy is close (0.93 vs target 0.95).

#5  Storm Runner  —  Voltline
    Genre: rock  |  Mood: intense  |  Energy: 0.91
    Score: 2.82 / 5.00
    Why:   energy is close (0.91 vs target 0.95).

=======================================================
```
