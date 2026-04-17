from engine.rules import CONDITIONS, evaluate_rule
from engine.generalizer import evaluate_program


def search_rules(examples, base_programs):

    best_rule = None
    best_score = 0.0

    condition_names = list(CONDITIONS.keys())

    for cond in condition_names:

        for prog_a in base_programs:
            for prog_b in base_programs:

                rule = {
                    "condition": cond,
                    "true_program": prog_a,
                    "false_program": prog_b
                }

                score = evaluate_rule(rule, examples, scorer=score)

                if score > best_score:
                    best_score = score
                    best_rule = rule

    return best_rule, best_score