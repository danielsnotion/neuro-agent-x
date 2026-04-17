# engine/objects.py

import numpy as np
from collections import deque


def find_objects(grid):
    visited = np.zeros_like(grid, dtype=bool)
    objects = []

    h, w = grid.shape

    for i in range(h):
        for j in range(w):
            if visited[i, j]:
                continue

            color = grid[i, j]
            queue = deque([(i, j)])
            visited[i, j] = True

            coords = []

            while queue:
                x, y = queue.popleft()
                coords.append((x, y))

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < h and 0 <= ny < w:
                        if not visited[nx, ny] and grid[nx, ny] == color:
                            visited[nx, ny] = True
                            queue.append((nx, ny))

            objects.append({
                "color": color,
                "coords": coords
            })

    return objects

# engine/objects.py (add below)

def crop_object(grid, obj):
    coords = obj["coords"]

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return grid[min_x:max_x+1, min_y:max_y+1]