<!-- 3083dc1b-3fc4-4e5b-9411-097f23963a43 9c2cf7e0-e82e-431f-a497-cdf989489c8d -->
# Terrain Generation Refactor

### Overview
Introduce a layered noise pipeline to build height/moisture maps, assign biomes with enforced adjacency rules, and generate ridge-like mountain ranges that respect buffers from water.

### Todos
- noise-utils — Add helper functions in `game_engine.py` (or a small module) for generating repeatable fractal noise maps and computing normalized distance-to-water masks.
- height-biome-pipeline (depends: noise-utils) — Replace `_initialize_map` with a staged routine that creates height & moisture layers, carves water, allocates plains/forest via hierarchical probabilities, and keeps land contiguous.
- mountain-ridge-pass (depends: height-biome-pipeline) — Implement a ridge-growing algorithm using gradient-following random walks so mountains form linear chains set back from water.
- spawn-adjustments (depends: mountain-ridge-pass) — Update spawn selection to prioritise plains/forest tiles with adequate distance from water and mountains, ensuring unique start spots.