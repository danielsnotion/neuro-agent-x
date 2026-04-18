# engine/search_multi.py

import numpy as np

from engine.executor import apply_program
from engine.scorer import score
from engine.color import get_color_map, apply_color_map
from engine.synthesizer import generate_candidate_programs
from engine.neural_scorer import predict_score
from engine.library import add_program


def evaluate_program(program, examples):

    scores = []

    for inp, tgt in examples:
        try:
            pred = apply_program(inp, program)
        except Exception:
            return 0.0

        if pred.shape != tgt.shape:
            scores.append(0.0)
        else:
            scores.append(score(pred, tgt))

    return np.mean(scores)


def beam_search_multi(examples, beam_width=10, max_depth=3):

    beam = [([], 0.0)]

    best_program = []
    best_score = 0.0

    input_grid, target_grid = examples[0]
    color_mappings = get_color_map(input_grid, target_grid)

    def program_complexity(p):
        return len(p)

    def uses_color_map(p):
        return any("color_map" in str(x) for x in p)

    for depth in range(max_depth):

        new_beam = []

        for program, _ in beam:

            # 🔥 neural-guided candidates
            candidates = generate_candidate_programs(program)

            candidates = sorted(
                candidates,
                key=lambda p: predict_score(p),
                reverse=True
            )[:20]

            for new_program in candidates:

                s = evaluate_program(new_program, examples)

                new_beam.append((new_program, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

            # 🔥 color mappings
            for mapping in color_mappings:
                new_program = program + [f"color_map:{mapping}"]

                s = evaluate_program(new_program, examples)

                new_beam.append((new_program, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

        # 🔥 sort
        new_beam = sorted(
            new_beam,
            key=lambda x: (
                x[1],
                not uses_color_map(x[0]),
                -program_complexity(x[0])
            ),
            reverse=True
        )

        beam = new_beam[:beam_width]

        print(f"[Multi] Depth {depth+1}: Best Score = {best_score}")

        if best_score == 1.0:
            break

    # store reusable programs
    if best_score > 0.9:
        add_program(best_program, best_score)

    return best_program, best_score