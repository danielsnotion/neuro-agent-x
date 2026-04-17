import numpy as np
from engine.executor import apply_program
from engine.scorer import score


def evaluate_program(program, examples):
    """
    examples = list of (input_grid, target_grid)
    """

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