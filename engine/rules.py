import numpy as np
from engine.executor import apply_program


# 🔹 CONDITIONS

def is_square(grid):
    return grid.shape[0] == grid.shape[1]


def has_multiple_colors(grid):
    return len(np.unique(grid)) > 1


def is_single_color(grid):
    return len(np.unique(grid)) == 1


def height_greater_than_width(grid):
    return grid.shape[0] > grid.shape[1]


def width_greater_than_height(grid):
    return grid.shape[1] > grid.shape[0]


CONDITIONS = {
    "is_square": is_square,
    "multi_color": has_multiple_colors,
    "single_color": is_single_color,
    "tall": height_greater_than_width,
    "wide": width_greater_than_height,
}


# 🔹 RULE ENGINE

def apply_rule(grid, rule):
    """
    rule = {
        "condition": "is_square",
        "true_program": [...],
        "false_program": [...]
    }
    """

    cond_fn = CONDITIONS[rule["condition"]]

    if cond_fn(grid):
        return apply_program(grid, rule["true_program"])
    else:
        return apply_program(grid, rule["false_program"])
    
def evaluate_rule(rule, examples, scorer):
    scores = []

    for inp, tgt in examples:
        try:
            pred = apply_rule(inp, rule)
        except Exception:
            return 0.0

        if pred.shape != tgt.shape:
            scores.append(0.0)
        else:
            scores.append(scorer(pred, tgt))

    return np.mean(scores)