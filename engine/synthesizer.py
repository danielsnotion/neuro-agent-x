from engine.transforms import TRANSFORMS
from engine.library import get_top_programs


def generate_candidate_programs(base_program):

    candidates = []

    # 🔹 extend program
    for t in TRANSFORMS.keys():
        candidates.append(base_program + [t])

    # 🔥 reuse learned programs
    learned = get_top_programs()

    for prog in learned:
        candidates.append(base_program + prog)

    return candidates