PROGRAM_LIBRARY = []


def add_program(program, score):
    PROGRAM_LIBRARY.append({
        "program": program,
        "score": score
    })


def get_top_programs(k=5):
    sorted_lib = sorted(PROGRAM_LIBRARY, key=lambda x: x["score"], reverse=True)
    return [p["program"] for p in sorted_lib[:k]]