DATASET = []


def add_sample(features, program, score):
    DATASET.append({
        "features": features,
        "program": program,
        "score": score
    })


def get_dataset():
    return DATASET