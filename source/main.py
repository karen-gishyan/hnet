from collections import deque, defaultdict
from utils import merge_dfs
from search import Graph, store_sequence, find_path, store_graph, find_path_graph

if __name__ == "__main__":
    df = merge_dfs(drug_name='PNEUMONIA')
    graph = Graph(df)
    # return list of sequences as lists, list of edges as dicts
    sequence_list, path_list = graph.search_space()

    ### graph
    edge_list_average_len = round(sum([len(dict_) for dict_ in path_list]) / len(path_list))
    adjacency_df, edges_cost_sum, start_edges_frequency, start_edges_cost_sum = store_graph(path_list)
    max_start_edge = max(start_edges_frequency, key=lambda _key: start_edges_frequency[_key])
    path_list = []
    explored = defaultdict(int)
    que = deque()
    que.append(max_start_edge[1])
    path_list.append(max_start_edge[1])
    explored[max_start_edge[1]] = 1
    find_path_graph(que, adjacency_df, edge_list_average_len, path_list, explored, edges_cost_sum)

    ### sequence
    sequence_list_average_len = round(sum([len(list_) for list_ in sequence_list]) / len(sequence_list))
    adjacency_df, start_edges_frequency = store_sequence(sequence_list)
    max_start_edge = max(start_edges_frequency, key=lambda _key: start_edges_frequency[_key])
    explored = defaultdict(int)
    path_list = []
    que = deque()
    que.append(max_start_edge[1])
    path_list.append(max_start_edge[1])
    explored[max_start_edge[1]] = 1
    find_path(que, adjacency_df, sequence_list_average_len, path_list, explored)
