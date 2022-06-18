import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from utils import calculate_date_intersection
from collections import Counter, defaultdict
from itertools import chain
import math


class Graph():
    def __init__(self, diagnosis_df: pd.DataFrame = None):
        self.graph = nx.DiGraph()
        self.df = diagnosis_df
        self.unique_admissions = list(diagnosis_df.hadm_id.unique())

    def search_space(self):
        """
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
         """
        # total count of drug pairs from each admission
        edge_list = []
        # total count of
        sequence_list = []
        for adm_i, admission in enumerate(self.unique_admissions):
            patient_df = self.df[self.df.hadm_id == admission]
            patient_df.reset_index(inplace=True)
            patient_stay_length = (patient_df.dischtime[0] - patient_df.admittime[0]).days + 1
            patient_df_edges = {}
            patient_df_sequence = []
            n_drugs = patient_df.shape[0]
            discharge_location = list(patient_df.discharge_location)[0]
            for i, row_i in patient_df.iterrows():

                if i <= n_drugs - 2:
                    child_row_1 = patient_df.iloc[i + 1]
                    # prevents self relationships
                    if row_i.drug != child_row_1.drug:
                        child_node_1_days_intersection = calculate_date_intersection(row_i, child_row_1)
                        # the more the intersection days the lesser the cost
                        if child_node_1_days_intersection:
                            relative_cost = round(patient_stay_length / child_node_1_days_intersection)
                        else:
                            relative_cost = patient_stay_length
                        child1_cost = self.cost_function(relative_cost, n_drugs, discharge_location)

                        # append (node,successor node) tuple to the sequence list
                        patient_df_sequence.append((row_i.drug, child_row_1.drug))
                        # append (node,successor node) tuple to the sequence list along with cost
                        patient_df_edges.update({(row_i.drug, child_row_1.drug): child1_cost})

                if i <= n_drugs - 3:
                    child_row_2 = patient_df.iloc[i + 2]
                    if row_i.drug != child_row_2.drug:
                        child_node_2_days_intersection = calculate_date_intersection(row_i, child_row_2)
                        if child_node_2_days_intersection:
                            # TODO think if this should be divided or not
                            # Should it be as a strengh(more the better) or cost (lower the better)
                            relative_cost = round(patient_stay_length / child_node_2_days_intersection) + 1
                        else:
                            # +1 cost for being not the directly connected node in the sequence
                            relative_cost = patient_stay_length + 1
                        child2_cost = self.cost_function(relative_cost, n_drugs, discharge_location)
                        patient_df_edges.update({(row_i.drug, child_row_2.drug): child2_cost})

            sequence_list.append(patient_df_sequence)
            edge_list.append(patient_df_edges)
            print(f"Iter {adm_i} completed")
        return sequence_list, edge_list

    def cost_function(self, days_intersection_cost: int, path_lengh: int, discharge_location: str):
        """
        Return the cost based on intersection days, discharge_outcome and path_length.
        :param days_intersection:
        :param discharge_location:
        :param path_lengh:
        :return:
        """
        if discharge_location == "DEAD/EXPIRED":
            return days_intersection_cost + math.log2(path_lengh) + 1  # fixed penalty
        return round(days_intersection_cost + math.log2(path_lengh))

    def visualize_sequence(self, sequence_list, index, subset=None):
        self.graph.add_edges_from(sequence_list[index][:subset])
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def visualize_graph(self, path_list, index, subset=None):
        weighted_edges = [(key[0], key[1], value) for key, value in
                          dict(list(path_list[index].items())[:subset]).items()]
        self.graph.add_weighted_edges_from(weighted_edges)
        nx.draw(self.graph, with_labels=True)
        plt.show()


def store_sequence(path_list2d):
    """
    Parameters:
        path_list2d: list of edges from each sequence, result of Graph().search_space()
    """
    # store starting edges from each path
    # for path in args, iterate over the path and store the first one
    start_edges = sorted([tuple_ for path in list(path_list2d) for (i, tuple_) in enumerate(path) if i == 0])
    # TODO if single listed is passed, no need to combine.
    # unpack the paths and chain them into a sorted list.
    path_combine = sorted(list(chain(*path_list2d)))
    # count the frequency of edges among start edges and total path edges
    start_edges_counter = Counter(start_edges)
    path_combine_counter = Counter(path_combine)
    # obtain distinct nodes from list of edge tuples
    node_list = sorted(set(list(chain.from_iterable(path_combine))))

    sorted_path_frequency = dict(sorted(path_combine_counter.items(), key=lambda t: t[1]))
    max_key_path_frequency = max(path_combine_counter, key=path_combine_counter.get)

    adjacency_df = pd.DataFrame(index=node_list, columns=node_list)
    adjacency_df.fillna(0, inplace=True)
    for key, value in path_combine_counter.items():
        adjacency_df.at[key[0], key[1]] = value
    return adjacency_df, start_edges_counter


def store_graph(list2d):
    """
     Parameters:
         path_list2d: list of dictionaries, where each dict stores edge pairs for each admission.
     """
    # start_edges = sorted([tuple_ for dict_ in list(list2d) for (i, tuple_) in enumerate(dict_.keys()) if i == 0])
    start_edges_frequency = defaultdict(int)
    start_edges_cost_sum = defaultdict(int)
    edges_frequency = defaultdict(int)
    edges_cost_sum = defaultdict(int)
    start_node_list, node_list = [], []
    for dict_ in list2d:
        for i, tuple_ in enumerate(dict_.keys()):
            # append each drug, uniqueness will be sorted later with set
            node_list.append(tuple_[0])
            node_list.append(tuple_[1])
            if i == 0:
                start_node_list.append(tuple_[0])
                start_node_list.append(tuple_[1])
                start_edges_frequency[tuple_] += 1
                start_edges_cost_sum[tuple_] += dict_[tuple_]
            # TODO think if this needs to be part of else
            edges_frequency[tuple_] += 1
            edges_cost_sum[tuple_] += dict_[tuple_]
    node_list = sorted(set(node_list))

    # show the sorted edges_frequency and the max_key
    sorted_edges_frequency = dict(sorted(edges_frequency.items(), key=lambda t: t[1]))
    max_key_edges_frequency = max(edges_frequency, key=edges_frequency.get)
    adjacency_df = pd.DataFrame(index=node_list, columns=node_list)
    adjacency_df.fillna(0, inplace=True)
    for key, value in edges_frequency.items():
        adjacency_df.at[key[0], key[1]] = value

    return adjacency_df, edges_cost_sum, start_edges_frequency, start_edges_cost_sum


def check_explored(**kwargs):
    """
    :param recursive_f: check_path or check_path_graph
    :param kwargs: kwargs (args) of check_path or check_path_graph
    :return:
    """
    # only has to know about a few kwargs, the rest are passed to the recursive_f.
    exit = False
    next_node_successor = None
    next_node = kwargs.get('next_node')
    explored = kwargs.get('explored')
    path_list = kwargs.get('path_list')
    que = kwargs.get('que')
    inside_look_ahead = kwargs.get('inside_look_ahead')

    if next_node not in explored:
        # if inside_look_ahead not passed (is None), append to the que
        if not inside_look_ahead: que.append(next_node)
        path_list.append(next_node)
        explored[next_node] += 1
    # if explored once, we allow to be part of the path only twice
    elif explored[next_node] <= 10:
        # if next_node (candidate node) has been previously explored, pass its successor nodes
        # to the recursive_f so as they are removed for the max calculation, otherwise will result in a loop
        # in the sequence
        next_node_indices = [i for i, node in enumerate(path_list) if node == next_node]
        next_node_successor = [path_list[i + 1] for i in next_node_indices]
        que.append(next_node)
        path_list.append(next_node)
        explored[next_node] += 1
    else:
        exit = True

    return exit, next_node_successor


def find_path(que, adjacency_matrix, average_path_length, path_list, explored,
              next_node_successor=None):
    """
    Search the graph for an optimal path by applying a recursive one step look ahead logic on the
    adjacency matrix generated from store graph.
    :return: path_list
    """
    if len(path_list) >= average_path_length:
        return path_list
    if not len(que):
        return path_list

    node = que.pop()
    row = adjacency_matrix.loc[node]
    # remove from max calculation
    if next_node_successor:
        row.drop(labels=next_node_successor, inplace=True)
        next_node_successor = None  # None to use it when needed and to prevent always entering here
    row_max = max(row)
    bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
    successor_list = bool_row.index[bool_row].tolist()
    if not successor_list:
        return path_list
    if len(successor_list) == 1:
        next_node = successor_list[0]
        exit, next_node_successor = check_explored(next_node=next_node,
                                                   explored=explored, que=que, path_list=path_list,
                                                   adjacency_matrix=adjacency_matrix,
                                                   average_path_length=average_path_length)
        # https://python-forum.io/thread-31275.html
        if exit:
            return path_list
    else:
        # perform a one step look ahead
        look_ahead_df = {}
        for n in successor_list:
            row = adjacency_matrix.loc[n]
            row_max = max(row)
            bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
            n_successor_list = bool_row.index[bool_row].tolist()
            for s in n_successor_list:
                look_ahead_df.update({(n, s): row_max})
        # select the first max appearing key
        # here max_nodes is a tuple instead of a single node
        # select the first maximum appearing node if there are multiple
        max_nodes = max(look_ahead_df, key=lambda _key: look_ahead_df[_key])
        # intermediary node during the look ahead should not be appended to the que,
        # for this reasons we pass inside look ahead
        next_node = max_nodes[0]
        exit, next_node_successor = check_explored(next_node=next_node,
                                                   explored=explored, que=que, path_list=path_list,
                                                   adjacency_matrix=adjacency_matrix,
                                                   average_path_length=average_path_length,
                                                   inside_look_ahead=True)
        if exit:
            return path_list
        next_node = max_nodes[1]
        exit, next_node_successor = check_explored(next_node=next_node,
                                                   explored=explored, que=que, path_list=path_list,
                                                   adjacency_matrix=adjacency_matrix,
                                                   average_path_length=average_path_length)
        if exit:
            return path_list
    # recursive step
    return find_path(que, adjacency_matrix, average_path_length, path_list, explored, next_node_successor)


def find_path_graph(que, adjacency_matrix, average_path_length, path_list, explored, edges_cost_sum,
                    next_node_successor=None):
    """
    Search the graph for an optimal path by applying a recursive one step look ahead logic on the
    adjacency matrix generated from store graph.
    :return: path_list
    """
    if len(path_list) >= average_path_length:
        return path_list
    if not len(que):
        return path_list

    node = que.pop()
    row = adjacency_matrix.loc[node]
    if next_node_successor:
        row.drop(labels=next_node_successor, inplace=True)
        next_node_successor = None  # None to use it when needed and to prevent always entering here
    row_max = max(row)
    bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
    successor_list = bool_row.index[bool_row].tolist()
    if not successor_list:
        return path_list
    if len(successor_list) == 1:
        next_node = successor_list[0]
        exit, next_node_successor = check_explored(next_node=next_node,
                                                   explored=explored, que=que, path_list=path_list,
                                                   adjacency_matrix=adjacency_matrix,
                                                   average_path_length=average_path_length,
                                                   edges_cost_sum=edges_cost_sum)
        if exit:
            return path_list
    else:
        # for graphs costs are also taken into account
        # another extra check to select the node with the maximum cost if successor list contains
        # multiple nodes
        assert edges_cost_sum, 'costs should be provided'
        successor_costs = {(node, n): edges_cost_sum.get((node, n)) for n in successor_list}
        successor_costs_list = [key[1] for key in successor_costs.keys()
                                if successor_costs[key] == min(successor_costs.values())]
        if len(successor_costs_list) == 1:
            next_node = successor_list[0]
            exit, next_node_successor = check_explored(next_node=next_node,
                                                       explored=explored, que=que, path_list=path_list,
                                                       adjacency_matrix=adjacency_matrix,
                                                       average_path_length=average_path_length,
                                                       edges_cost_sum=edges_cost_sum)
            if exit:
                return path_list
        else:
            # perform a one step look ahead
            look_ahead_df = {}
            for n in successor_list:
                row = adjacency_matrix.loc[n]
                row_max = max(row)
                bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
                n_successor_list = bool_row.index[bool_row].tolist()
                print('n_successor_list', n_successor_list)
                for s in n_successor_list:
                    look_ahead_df.update({(n, s): row_max})
            max_nodes = max(look_ahead_df, key=lambda _key: look_ahead_df[_key])
            next_node = max_nodes[0]
            exit, next_node_successor = check_explored(next_node=next_node,
                                                       explored=explored, que=que, path_list=path_list,
                                                       adjacency_matrix=adjacency_matrix,
                                                       average_path_length=average_path_length,
                                                       edges_cost_sum=edges_cost_sum,
                                                       inside_look_ahead=True)
            if exit:
                return path_list
            next_node = max_nodes[1]
            exit, next_node_successor = check_explored(next_node=next_node,
                                                       explored=explored, que=que, path_list=path_list,
                                                       adjacency_matrix=adjacency_matrix,
                                                       average_path_length=average_path_length,
                                                       edges_cost_sum=edges_cost_sum)
            if exit:
                return path_list
    # recursive step
    return find_path_graph(que, adjacency_matrix, average_path_length, path_list, explored, edges_cost_sum,
                           next_node_successor)



