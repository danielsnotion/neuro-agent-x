# engine/search.py

import numpy as np
import time

from engine.executor import apply_program
from engine.scorer import score
from engine.color import get_color_map, apply_color_map
from engine.synthesizer import generate_candidate_programs
from engine.library import add_program
from engine.neural_scorer import predict_score

from models.dataset import add_sample
from engine.features import extract_features, extract_object_features


def beam_search(
    input_grid,
    target_grid,
    beam_width=20,
    max_depth=7,
    use_neural=True,
    time_limit=2.0
):

    start_time = time.time()

    beam = [([], input_grid, 0.0)]
    best_program = []
    best_score = 0.0

    color_mappings = get_color_map(input_grid, target_grid)

    # 🔧 feature extraction (for dataset)
    f1 = extract_features(input_grid)
    f2 = extract_object_features(input_grid)
    features = {**f1, **f2}

    # 🔧 helpers
    def is_same(a, b):
        return a.shape == b.shape and np.array_equal(a, b)

    def program_complexity(p):
        return len(p)

    def uses_color_map(p):
        return any("color_map" in str(x) for x in p)

    # 🔥 cache to avoid recomputation
    visited_states = set()

    for depth in range(max_depth):

        # ⏱️ time limit check
        if time.time() - start_time > time_limit:
            print("⏱️ Time limit reached")
            break

        new_beam = []

        for program, current_grid, _ in beam:

            # 🔥 generate candidates
            candidates = generate_candidate_programs(program)

            # 🔥 prune candidate size early
            candidates = candidates[:50]

            # 🔥 neural ranking
            if use_neural:
                candidates = sorted(
                    candidates,
                    key=lambda p: predict_score(p, input_grid),
                    reverse=True
                )[:10]  # strong pruning
            else:
                candidates = candidates[:10]

            for new_program in candidates:

                try:
                    new_grid = apply_program(input_grid, new_program)
                except Exception:
                    continue

                # 🔥 prune invalid size
                if new_grid.shape[0] > 30 or new_grid.shape[1] > 30:
                    continue

                # 🔥 skip no-op
                if is_same(new_grid, input_grid):
                    continue

                # 🔥 caching (avoid duplicate states)
                key = new_grid.tobytes()
                if key in visited_states:
                    continue
                visited_states.add(key)

                # 🔥 scoring
                if new_grid.shape != target_grid.shape:
                    s = 0.0
                else:
                    s = score(new_grid, target_grid)

                # 🔥 dataset collection
                add_sample(features, new_program, s)

                new_beam.append((new_program, new_grid, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

            # 🔥 color mapping branch
            for mapping in color_mappings:

                try:
                    new_grid = apply_color_map(current_grid, mapping)
                except Exception:
                    continue

                if is_same(new_grid, current_grid):
                    continue

                key = new_grid.tobytes()
                if key in visited_states:
                    continue
                visited_states.add(key)

                if new_grid.shape != target_grid.shape:
                    s = 0.0
                else:
                    s = score(new_grid, target_grid)

                new_program = program + [f"color_map:{mapping}"]

                add_sample(features, new_program, s)

                new_beam.append((new_program, new_grid, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

        # 🔥 beam ranking
        new_beam = sorted(
            new_beam,
            key=lambda x: (
                x[2],
                not uses_color_map(x[0]),
                -program_complexity(x[0])
            ),
            reverse=True
        )

        beam = new_beam[:beam_width]

        print(
            f"[Depth {depth+1}] Beam={len(beam)} | Best Score={best_score}"
        )

        # early stop
        if best_score == 1.0:
            break

    # 🔥 fallback safety
    if not best_program and beam:
        best_program = beam[0][0]
        best_score = beam[0][2]

    # 🔥 store reusable programs
    if best_score > 0.9:
        add_program(best_program, best_score)

    return best_program, best_score