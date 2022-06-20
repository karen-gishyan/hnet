from collections import deque, defaultdict
from utils import merge_dfs, evaluate
from search import Graph, store_sequence, find_path, store_graph, find_path_graph
import logging

if __name__ == "__main__":
    # configure a logger
    logger = logging.getLogger('hnet')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('info.log')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    df = merge_dfs(drug_name='PNEUMONIA')
    graph = Graph(df)
    # return list of sequences as lists, list of edges as dicts
    sequence_list, path_list = graph.search_space(number_of_admissions=5)
    graph.visualize_sequence(sequence_list, 0, subset=11)
    graph.visualize_graph(path_list, 0, subset=12)

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

    path_graph = find_path_graph(que, adjacency_df, edge_list_average_len, path_list, explored, edges_cost_sum)
    logger.info(f'The path for graph is {path_graph}')
    print(f'The path for graph is {path_graph}')
    # TODO evaluate how more admissions affect the evaluation score
    score = evaluate(path_graph, graph, number_of_admissions=3)
    logger.info(f'The ratcliff_obershelp score is {score}')
    print(f'The ratcliff_obershelp score is {score}')

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

    path = find_path(que, adjacency_df, sequence_list_average_len, path_list, explored)
    print(f'The path is {path}')
    logger.info(f'The path is {path}')
    score = evaluate(path, graph, number_of_admissions=3)
    print(f'The ratcliff_obershelp score is {score}')
    logger.info(f'The ratcliff_obershelp score is {score}')
