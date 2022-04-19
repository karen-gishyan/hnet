from collections import deque
from source.utils import make_train_data, get_drug_weights, merge_dfs
from source.search import Graph, store_graph, find_path

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
    df = merge_dfs(drug_name='PNEUMONIA')
    train_df = make_train_data(df)
    drug_weights = get_drug_weights(train_df)

    graph = Graph(df)
    # TODO issue in search_space
    # graph.search_space().visualize()
