from source.utils import model_preprocess, svc, diagnosis_value_counts, merge_admissions_prescriptions
from source.search import Graph
import pandas as pd

if __name__ == "__main__":
    df = merge_admissions_prescriptions(diagnosis_value_counts())
    # df=pd.read_csv('data/admissions_prescriptions_merge.csv')
    unique_diagnosis = df.diagnosis.unique()
    # construct a search tree for each unique_diagnosis
    for diagnosis in unique_diagnosis:
        diagnosis_df = df[df.diagnosis == diagnosis]
        Graph(diagnosis_df).between_node_heuristic().visualize()
        # break
        # weights = svc(model_preprocess())
        # Graph().node_i_to_end_node_heuristic(weights)
