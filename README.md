# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Permana61/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Permana61/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python forecaster that predicts the earn rate of every GTA VI trophy on Exophase one week after release. Round 2 rebuilds the model around how Exophase actually works: the page is a frozen snapshot between refreshes, so the forecast predicts the *true* tracked-player count and the *true* earners at the target, not the stale page. A flat history means no refresh has happened yet, so the player count follows the launch curve (638 tracked players to 3,868 at the first refresh, 4.5 hours after release) and fresh syncers dilute every rate except the opening mission, which every new player finishes. A moving history is fitted directly: per-trophy elasticity of earners against players, shrunk toward a difficulty prior from tier, points and description, with a logit trend as a cross-check. Quantiles are logit-normal with a spread that widens with horizon, fit residual and small earner counts, and medians are pooled so the story chain, the 100%-completion trophy and the strand finals stay in their logical order. Built for the make.fun $250 bounty.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=3zcdv8qs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
