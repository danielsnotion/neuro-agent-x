from engine.transforms import TRANSFORMS
from engine.color import get_color_map, apply_color_map
from engine.generalizer import evaluate_program
from engine.selector import get_initial_programs

def beam_search_multi(examples, beam_width=10, max_depth=3):

    initial_programs = get_initial_programs([(input_grid, target_grid)])

    beam = [(p, input_grid, 0.0) for p in initial_programs] or [([], input_grid, 0.0)]

    best_program = []
    best_score = 0.0

    transform_names = list(TRANSFORMS.keys())

    # use first example for color mapping
    input_grid, target_grid = examples[0]
    color_mappings = get_color_map(input_grid, target_grid)

    def program_complexity(program):
        return len(program)

    def uses_color_map(program):
        return any("color_map" in str(p) for p in program)

    for depth in range(max_depth):

        new_beam = []

        for program, _ in beam:

            # 🔹 geometric transforms
            for t in transform_names:
                new_program = program + [t]

                s = evaluate_program(new_program, examples)

                new_beam.append((new_program, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

            # 🔥 color mappings
            for mapping in color_mappings:
                new_program = program + [f"color_map:{mapping}"]

                s = evaluate_program(new_program, examples)

                new_beam.append((new_program, s))

                if s > best_score:
                    best_score = s
                    best_program = new_program

        # 🔥 sort programs
        new_beam = sorted(
            new_beam,
            key=lambda x: (
                x[1],                      # score
                not uses_color_map(x[0]),
                -program_complexity(x[0])
            ),
            reverse=True
        )

        beam = new_beam[:beam_width]

        print(f"Depth {depth+1}: Best Score = {best_score}")

        if best_score == 1.0:
            break

    return best_program, best_score