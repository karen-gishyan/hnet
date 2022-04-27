import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from source.utils import calculate_date_intersection
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
        edge_list = []  # total list of edges from each admission
        sequence_list = []  # drugs as sequences, duplicates removed
        for adm_i, admission in enumerate(self.unique_admissions):
            patient_df = self.df[self.df.hadm_id == admission]
            patient_df.reset_index(inplace=True)
            patient_stay_length = (patient_df.dischtime[0] - patient_df.admittime[0]).days + 1
            patient_df_edges = {}
            patient_df_sequence = []
            n_drugs = patient_df.shape[0]
            discharge_location = list(patient_df.discharge_location)[0]
            for i, row_i in patient_df.iterrows():
                if i == n_drugs - 3:  # check this part to again
                    break
                child_row_1, child_row_2 = patient_df.iloc[i + 1], patient_df.iloc[i + 2]

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

                if row_i.drug != child_row_2.drug:
                    child_node_2_days_intersection = calculate_date_intersection(row_i, child_row_2)
                    if child_node_2_days_intersection:
                        relative_cost = round(patient_stay_length / child_node_2_days_intersection) + 1
                    else:
                        # +1 cost for being not the directly connected node in the sequence
                        relative_cost = patient_stay_length + 1
                    child2_cost = self.cost_function(relative_cost, n_drugs, discharge_location)
                    patient_df_edges.update({(row_i.drug, child_row_2.drug): child2_cost})

            sequence_list.append(patient_df_sequence)
            edge_list.append(patient_df_edges)
            print(f"Iter {adm_i} completed")
            if adm_i == 2:
                break
        # print(nx.is_tree(self.graph))
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

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        # labels = nx.get_edge_attributes(self.graph, 'cost')
        # labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        # nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
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
    # start_node_list = sorted(set(start_node_list))

    adjacency_df = pd.DataFrame(index=node_list, columns=node_list)
    adjacency_df.fillna(0, inplace=True)
    # start_adjacency_df = pd.DataFrame(index=start_node_list, columns=start_node_list)
    # start_adjacency_df.fillna(0, inplace=True)

    for key, value in edges_frequency.items():
        adjacency_df.at[key[0], key[1]] = value
    # for key, value in start_edges_frequency.items():
    #     start_adjacency_df.at[key[0], key[1]] = value

    return adjacency_df, edges_cost_sum, start_edges_frequency, start_edges_cost_sum


def check_explored(next_node, explored, que, path_list, adjacency_matrix, average_path_length):
    """
    Includes a recusrive proportion, part of find_path.
    :param next_node:
    :param explored:
    :param que:
    :param path_list:
    :param adjacency_matrix:
    :param average_path_length:
    :return:
    """
    if next_node not in explored:
        que.append(next_node)
        path_list.append(next_node)
        explored[next_node] += 1
    # if explored once, we allow to be part of the path only once
    elif explored[next_node] == 1:
        next_node_index = path_list.index(next_node)
        next_node_successor = path_list[next_node_index + 1]
        que.append(next_node)
        path_list.append(next_node)
        explored[next_node] += 1
        # TODO why does it not complain about find_path(), ok, it is define in search.py
        find_path(que, adjacency_matrix, average_path_length, path_list, explored,
                  next_node_successor)  # next_node_successor is added
    else:
        exit = True
        return exit


global_exit = False

def find_path(que, adjacency_matrix, average_path_length, path_list, explored,
              next_node_successor=None, exit=False):
    """
    Search the graph for an optimal path by applying a recursive one step look ahead logic on the
    adjacency matrix generated from store graph.
    :return: path_list
    """
    # TODO short term solution, needs to change
    global global_exit
    if global_exit:
        print('The path is', path_list)
        return path_list
    #TODO logic may need to change
    if len(path_list) >= average_path_length:
        print('The path is', path_list)
        que.clear()
        return path_list
    if not len(que):
        print('The final path for sequence is', path_list)
        return path_list

    node = que.pop()
    row = adjacency_matrix.loc[node]
    row_max = max(row)
    bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
    successor_list = bool_row.index[bool_row].tolist()
    if next_node_successor:
        successor_list.remove(next_node_successor)
        next_node_successor = None  # None to use it when needed and to prevent always entering here
    if not successor_list:
        return path_list
    if len(successor_list) == 1:
        next_node = successor_list[0]
        exit = check_explored(next_node, explored, que, path_list, adjacency_matrix, average_path_length)
        if exit:
            global_exit=True
            return
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
        # select the first max appearing key
        # here max_nodes is a tuple instead of a single node
        max_nodes = max(look_ahead_df, key=lambda _key: look_ahead_df[_key])
        next_node = max_nodes[0]
        # TODO here exit return a pointer the function, function is reentered.
        # TODO why does it then go to the elif of check_explored() function.

        exit = check_explored(next_node, explored, que, path_list, adjacency_matrix, average_path_length)
        if exit:
            global_exit = True
            return
        next_node = max_nodes[1]
        exit = check_explored(next_node, explored, que, path_list, adjacency_matrix, average_path_length)
        if exit:
            global_exit = True
            return
    find_path(que, adjacency_matrix, average_path_length, path_list, explored, exit)


def find_path_graph(que, adjacency_matrix, average_path_length, path_list, explored, edges_cost_sum):
    """
    Search the graph for an optimal path by applying a recursive one step look ahead logic on the
    adjacency matrix generated from store graph.
    :return: path_list
    """
    # breakpoint condition
    # TODO the breakpoint logic may need to be more complex
    if len(path_list) >= average_path_length:
        print('The path is', path_list)
        que.clear()
        return path_list
    if not len(que):
        print('The final path for graph is', path_list)
        return path_list
    else:
        node = que.pop()
        row = adjacency_matrix.loc[node]
        row_max = max(row)
        bool_row = row.apply(lambda val: val == row_max if row_max != 0 else False)
        successor_list = bool_row.index[bool_row].tolist()
        print('successor_list', successor_list)
        # breakpoint condition, return if no successor
        # TODO the breakpoint logic may need to be more complex
        if not successor_list:
            print('The path is', path_list)
            que.clear()
            return path_list
        if len(successor_list) == 1:
            if successor_list[0] not in explored:
                que.append(successor_list[0])
                path_list.append(successor_list[0])
                explored.append(successor_list[0])
            else:
                # TODO logic may need to be more complex
                return path_list
        else:
            # for graphs costs are also taken into account
            assert edges_cost_sum, 'costs should be provided'
            successor_costs = {(node, n): edges_cost_sum.get((node, n)) for n in successor_list}
            successor_costs_list = [key for key in successor_costs.keys()
                                    if successor_costs[key] == max(successor_costs.values())]
            if len(successor_costs_list) == 1:
                if successor_costs_list[0][1] not in explored:
                    que.append(successor_costs_list[0][1])
                    path_list.append(successor_costs_list[0][1])
                    explored.append(successor_costs_list[0][1])
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
                # select the first max appearing key
                # TODO issue ('insulin -> ipratropium -> insulin) and repeating starts
                max_node = max(look_ahead_df, key=lambda _key: look_ahead_df[_key])
                path_list.append(max_node[0])
                if max_node[1] not in explored:
                    que.append(max_node[1])
                    path_list.append(max_node[1])
                    explored.append(max_node[1])
    find_path_graph(que, adjacency_matrix, average_path_length, path_list, explored, edges_cost_sum)
