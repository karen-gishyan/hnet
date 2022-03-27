from source.utils import model_preprocess, svc, diagnosis_value_counts, merge_admissions_prescriptions
from source.search import Graph

if __name__ == "__main__":
    df = merge_admissions_prescriptions(diagnosis_value_counts(value_count=2))
    unique_diagnosis = df.diagnosis.unique()
    # construct a search tree for each unique_diagnosis
    for diagnosis in unique_diagnosis:
        diagnosis_df = df[df.diagnosis == diagnosis]
        graph = Graph(diagnosis_df)
        graph.start_end_to_drug_cost()
        graph.between_node_cost()
        break
        # weights = svc(model_preprocess())
        # Graph().node_i_to_end_node_heuristic(weights)
