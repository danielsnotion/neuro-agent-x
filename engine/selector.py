from engine.meta import retrieve_similar
from engine.features import extract_task_features


def get_initial_programs(examples):

    features = extract_task_features(examples)[0]

    similar_programs = retrieve_similar(features)

    return similar_programs