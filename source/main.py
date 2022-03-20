from source.utils import model_preprocess, svc, diagnosis_value_counts, merge_admissions_prescriptions
from source.search import Graph

if __name__ == "__main__":

    df = merge_admissions_prescriptions(diagnosis_value_counts(value_count=2))
    unique_diagnosis = df.diagnosis.unique()
    # construct a search tree for each unique_diagnosis
    for diagnosis in unique_diagnosis:
        diagnosis_df = df[df.diagnosis == diagnosis]
        Graph(diagnosis_df).between_node_cost().bfs().visualize()
        break
        # weights = svc(model_preprocess())
        # Graph().node_i_to_end_node_heuristic(weights)
