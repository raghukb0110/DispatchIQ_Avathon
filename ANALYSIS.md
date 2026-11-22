# Algorithm Comparison: Hungarian (batch) vs ML Dispatcher (online)

The numbers come from `python benchmark.py --seeds 5` (5 seeds per preset, default weights) and from the held-out
evaluation in [`backend/models/model_card.json`](backend/models/model_card.json) (40 unseen random days).
Everything is reproducible with the same seeds.

## Results per scenario type

| Preset | Algorithm | Fulfilled % | Priority-weighted % | High-priority % | km / job | Idle min | Workload σ | ms (whole day) |
|---|---|---|---|---|---|---|---|---|
| balanced | Hungarian | 83.0 | **86.1** | **89.5** | 8.37 | 44.9 | **0.76** | 5 |
| balanced | ML | **84.0** | 83.8 | 84.4 | **8.33** | **39.6** | 1.15 | 101 |
| scarce | Hungarian | 50.4 | **60.1** | **80.1** | **6.84** | 20.8 | **0.87** | 7 |
| scarce | ML | **52.2** | 50.5 | 48.7 | 10.30 | **6.9** | 0.93 | 93 |
| clustered | Hungarian | 77.5 | **84.4** | **92.9** | **5.10** | 31.0 | **0.69** | 7 |
| clustered | ML | **80.8** | 80.6 | 78.0 | 5.98 | **23.3** | 1.03 | 131 |
| rare_skills | Hungarian | 73.1 | **78.9** | **92.0** | **8.53** | 36.7 | **0.91** | 10 |
| rare_skills | ML | **74.0** | 74.8 | 77.9 | 9.73 | **35.5** | 1.19 | 108 |
| large (80×600) | Hungarian | 72.8 | **84.2** | **97.9** | **3.04** | 12.2 | 0.67 | 380 |
| large (80×600) | ML | **75.9** | 76.0 | 75.3 | 3.99 | **5.3** | **0.51** | 957 |

**Agreement:** the share of jobs where both algorithms chose the same technician. It's 0.43 for balanced, 0.27 for
scarce, 0.37 for clustered, 0.44 for rare_skills and 0.14 for large.

**Imitation quality** (unseen days): the ML model picks the same technician as the optimiser in **60.2 %** of
non-trivial decisions. "Pick the cheapest technician" manages 40.9 %, and random choice 23.3 %.

## What I learned

1. **Batch optimisation wins on *who* gets served.** Hungarian sees every request at once, so urgent jobs win the
   competition for technicians. It serves 80–98 % of priority 4–5 jobs, against 49–84 % for the online dispatcher.
   The gap is biggest when capacity is scarce (`scarce`: 80 % vs 49 %).

2. **Online dispatch wins on *how many* get served.** The ML dispatcher fulfils 1–3 points more jobs in **all five**
   presets and has less idle time in all five. Hungarian solves in rounds, and each round makes every technician take
   its best job *for that round*, sometimes one late in the day. That leaves idle gaps and blocks earlier work.
   Deciding in request order fills schedules more densely.

3. **The model learned more than the cost formula.** Hungarian picks the cheapest technician in only about 40 % of
   decisions. The model reproduces its choice 60 % of the time. Its most important signals are slack before the
   deadline, distance, over-qualification, and **skill rarity**: it learned to keep scarce-skill technicians free,
   which Hungarian does implicitly. Removing the skill-scarcity features costs about 3 points of top-1 accuracy.

4. **Where ML struggles.** In 59 % of decisions Hungarian deliberately skipped the cheapest option. The model agrees
   with only 48 % of those overrides, because they depend on *future* jobs an online policy can't see. Agreement also
   falls with fleet size (0.45 → 0.12). One different early choice changes the state for every later decision, so the
   model drifts into situations it never saw in training (compounding error). Even so, its fulfilment stays above
   Hungarian's.

5. **Speed is not the argument.** SciPy's Hungarian solver is compiled code: a whole 80×600 day takes about 0.4 s.
   The ML dispatcher spends about 1.5–2.3 ms per request. Its value is **deciding immediately without waiting for a
   batch**, not raw throughput.

## When to use which

| Situation | Choice |
|---|---|
| Requests known in advance (next-day planning) | Hungarian / batch optimisation |
| Priority SLAs dominate | Hungarian |
| Requests arrive continuously and need an immediate answer | ML dispatcher |
| Heavy contention but some latency is acceptable | Rolling micro-batches (re-solve Hungarian every ~15 min) |

## Limitations and next steps

- **Synthetic data.** The model learns the generator's world. Real dispatch history would replace the teacher
  labels, with a time-based train/test split.
- **The teacher's ceiling.** Round-based Hungarian is not a full routing optimum (VRPTW). An OR-Tools VRPTW teacher
  would raise the ceiling and remove the idle-time habit the model copied.
- **DAgger** (label the states the model itself visits) should reduce compounding error at scale.
- **Priority-aware online policy.** Briefly hold capacity when high-priority demand is forecast.
