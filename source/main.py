from collections import deque

from source.utils import model_preprocess, svc, diagnosis_value_counts, merge_admissions_prescriptions
from source.search import Graph, store_graph, find_path
import pandas as pd

if __name__ == "__main__":
    #### path merging algorithm
    path1 = [('A', 'B'), ('B', 'C'), ('C', 'E')]
    path2 = [('A', 'C'), ('C', 'E'), ('E', 'D')]
    path3 = [('B', 'F'), ('F', 'G'), ('G', 'E'), ('E', 'F')]

    average_path_length = round((len(path1) + len(path2) + len(path3)) / 3)
    start_edges, adjacency_matrix = store_graph(path1, path1, path3)
    max_start_edge = max(start_edges, key=lambda _key: start_edges[_key])
    path_list = []
    que = deque()
    que.append(max_start_edge[1])
    path_list.append(max_start_edge[1])
    find_path(que, adjacency_matrix, average_path_length, path_list)

    #### A Star logic
    df = merge_admissions_prescriptions(diagnosis_value_counts(value_count=2))
    # df=pd.read_csv('data/val_2.csv')
    unique_diagnosis = df.diagnosis.unique()
    # construct a search tree for each unique_diagnosis
    for diagnosis in unique_diagnosis:
        diagnosis_df = df[df.diagnosis == diagnosis]
        graph = Graph(diagnosis_df)
        graph.search_space().visualize()
        break
