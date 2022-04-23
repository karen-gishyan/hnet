import os
import sys

cwd = os.getcwd()
if not cwd in sys.path:
    sys.path.append('.')

from collections import deque
from source.utils import make_train_data, get_drug_weights, merge_dfs
from source.search import Graph, store_sequence, find_path, store_graph

if __name__ == "__main__":
    #### path merging algorithm
    path1 = [('A', 'B'), ('B', 'C'), ('C', 'E')]
    path2 = [('A', 'C'), ('C', 'E'), ('E', 'D')]
    path3 = [('B', 'F'), ('F', 'G'), ('G', 'E'), ('E', 'F')]

    paths = [i for i in (path1, path2, path3)]
    average_path_length = round((len(path1) + len(path2) + len(path3)) / 3)
    start_edges, adjacency_matrix = store_sequence(paths)
    max_start_edge = max(start_edges, key=lambda _key: start_edges[_key])
    path_list = []
    que = deque()
    que.append(max_start_edge[1])
    path_list.append(max_start_edge[1])
    find_path(que, adjacency_matrix, average_path_length, path_list)

    ##########
    df = merge_dfs(drug_name='PNEUMONIA')
    graph = Graph(df)
    sequenc_list, edge_list = graph.search_space()
    adjacency_df, start_adjacency_df, edges_cost_sum, start_edges_cost_sum = store_graph(edge_list)
    adjacency_df.to_csv('data/adjacency_df.csv')
    start_adjacency_df.to_csv('data/start_adjacency_df.csv')
