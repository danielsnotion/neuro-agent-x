# engine/search.py

import numpy as np

from engine.transforms import TRANSFORMS
from engine.scorer import score
from engine.color import get_color_map, apply_color_map


def beam_search(input_grid, target_grid, beam_width=10, max_depth=3):

    # (program, current_grid, score)
    beam = [([], input_grid, 0.0)]

    best_program = []
    best_score = 0.0

    transform_names = list(TRANSFORMS.keys())

    # generate color mappings once
    color_mappings = get_color_map(input_grid, target_grid)

    # 🔧 helpers
    def program_complexity(program):
        return len(program)

    def uses_color_map(program):
        return any("color_map" in str(p) for p in program)

    def is_same(a, b):
        return a.shape == b.shape and np.array_equal(a, b)

    for depth in range(max_depth):

        new_beam = []

        for program, current_grid, _ in beam:

            # 🔹 geometric transforms
            for t in transform_names:
                try:
                    new_grid = TRANSFORMS[t](current_grid)
                except Exception:
                    continue

                # 🔥 PRUNING: skip invalid shapes
                if new_grid.shape[0] > 30 or new_grid.shape[1] > 30:
                    continue

                # 🔥 PRUNING: skip same state
                if is_same(new_grid, current_grid):
                    continue

                # 🔥 scoring
                if new_grid.shape != target_grid.shape:
                    s = 0.0
                else:
                    s = score(new_grid, target_grid)

                new_program = program + [t]

                new_beam.append((new_program, new_grid, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

            # 🔥 color mappings
            for mapping in color_mappings:
                try:
                    new_grid = apply_color_map(current_grid, mapping)
                except Exception:
                    continue

                if is_same(new_grid, current_grid):
                    continue

                if new_grid.shape != target_grid.shape:
                    s = 0.0
                else:
                    s = score(new_grid, target_grid)

                new_program = program + [f"color_map:{mapping}"]

                new_beam.append((new_program, new_grid, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

        # 🔥 GLOBAL PRUNING: remove duplicate states
        unique_states = []
        dedup_beam = []

        for prog, grid, s in new_beam:
            if not any(is_same(grid, g) for g in unique_states):
                unique_states.append(grid)
                dedup_beam.append((prog, grid, s))

        # 🔥 FINAL SORT (ONLY ONCE)
        dedup_beam = sorted(
            dedup_beam,
            key=lambda x: (
                x[2],                      # score
                not uses_color_map(x[0]), # prefer simpler transforms
                -program_complexity(x[0]) # shorter programs
            ),
            reverse=True
        )

        # keep top-k
        beam = dedup_beam[:beam_width]

        print(f"Depth {depth+1}: Best Score = {best_score}")

        # early stop
        if best_score == 1.0:
            break

    return best_program, best_score