META_DB = []


def store_experience(features, program, score):
    META_DB.append({
        "features": features,
        "program": program,
        "score": score
    })


def similarity(f1, f2):
    score = 0

    if f1["num_colors"] == f2["num_colors"]:
        score += 1
    if f1["is_square"] == f2["is_square"]:
        score += 1

    return score


def retrieve_similar(features, top_k=3):
    ranked = []

    for item in META_DB:
        sim = similarity(features, item["features"])
        ranked.append((sim, item["program"]))

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)

    return [p for _, p in ranked[:top_k]]