import json
import os

DATA_PATH = "models/dataset.json"


def add_sample(features, program, score):

    sample = {
        "features": features,
        "program": program,
        "score": float(score)
    }

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(sample)

    with open(DATA_PATH, "w") as f:
        json.dump(data, f)


def get_dataset():

    if not os.path.exists(DATA_PATH):
        return []

    with open(DATA_PATH, "r") as f:
        return json.load(f)