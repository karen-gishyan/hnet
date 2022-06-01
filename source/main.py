from collections import deque, defaultdict
from utils import merge_dfs
from search import Graph, store_sequence, find_path, store_graph, find_path_graph

if __name__ == "__main__":
    #### sample path merging algorithm
    # path1 = [('A', 'B'), ('B', 'C'), ('C', 'E')]
    # path2 = [('A', 'C'), ('C', 'E'), ('E', 'D')]
    # path3 = [('B', 'F'), ('F', 'G'), ('G', 'E'), ('E', 'F')]
    #
    # paths = [i for i in (path1, path2, path3)]
    # average_path_length = round((len(path1) + len(path2) + len(path3)) / 3)
    # adjacency_matrix, start_edges = store_sequence(paths)
    # max_start_edge = max(start_edges, key=lambda _key: start_edges[_key])
    # path_list = []
    # que = deque()
    # explored = defaultdict(int)
    # que.append(max_start_edge[1])
    # path_list.append(max_start_edge[1])
    # explored[max_start_edge[1]] = 1
    # find_path(que, adjacency_matrix, average_path_length, path_list,explored)
    
    ##########
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
    # TODO after this call, global is modified, then find path exits imediatly by checking global=True
    #  For now need to be run separately
    # find_path_graph(que, adjacency_df, edge_list_average_len, path_list, explored, edges_cost_sum)

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
    #TODO repeating prints need to be removed
    find_path(que, adjacency_df, sequence_list_average_len, path_list, explored)
