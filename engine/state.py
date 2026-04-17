import numpy as np


class GridState:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=np.int32)
        self.shape = self.grid.shape
        self.colors = set(np.unique(self.grid))

    def copy(self):
        return GridState(self.grid.copy())

    def __repr__(self):
        return f"GridState(shape={self.shape}, colors={self.colors})"