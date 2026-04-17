

from engine.transforms import TRANSFORMS


def apply_program(grid, program):
    """
    grid: numpy array
    program: list of transform names (strings)
    """

    current = grid.copy()

    for step in program:
        if step not in TRANSFORMS:
            raise ValueError(f"Unknown transform: {step}")

        current = TRANSFORMS[step](current)

    return current