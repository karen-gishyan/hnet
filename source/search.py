import json
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from source.utils import calculate_date_intersection
from collections import Counter
from itertools import chain


class Graph():
    if not os.path.isdir('jsons'): os.mkdir('jsons')

    def __init__(self, diagnosis_df: pd.DataFrame = None):
        self.graph = nx.DiGraph()
        self.df = diagnosis_df
        self.df_length = len(self.df)
        self.unique_admissions = list(diagnosis_df.hadm_id.unique())
        self.diagnosis_name = self.df.diagnosis[0]
        super(Graph, self).__init__()

    def search_space(self):
        """
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
         """
        for adm_i, admission in enumerate(self.unique_admissions):
            patient_df = self.df[self.df.hadm_id == admission]
            patient_df_heuristics = {}
            for i, row_i in patient_df.iterrows():
                if i == self.df_length - 2:  # check this part to again
                    break
                child_row_1, child_row_2 = patient_df.iloc[i + 1], patient_df.iloc[i + 2]
                child_node_1_days_intersection = calculate_date_intersection(row_i, child_row_1)
                child_node_2_days_intersection = calculate_date_intersection(row_i, child_row_2)
                patient_stay_length = (row_i.dischtime - row_i.admittime).days + 1
                if child_node_1_days_intersection:
                    child_node_1_cost = 1 - round(patient_stay_length / child_node_1_days_intersection)
                else:
                    child_node_1_cost = 1
                if child_node_2_days_intersection:
                    child_node_2_cost = 2 - round(patient_stay_length / child_node_2_days_intersection)
                else:
                    child_node_2_cost = 2
                self.graph.add_edge(row_i.drug, child_row_1.drug, cost=child_node_1_cost)
                self.graph.add_edge(row_i.drug, child_row_2.drug, cost=child_node_2_cost)

                # parent_h1 = self.consistent_heuristic(row_i.drug, child_row_1.drug, child_node_1_cost)
                # parent_h2 = self.consistent_heuristic(row_i.drug, child_row_2.drug, child_node_2_cost)
                # parent_h = min(parent_h1, parent_h2)
                # patient_df_heuristics.update({row_i: parent_h})

            # def heuristics_for_a_star(_, b):
            #     return patient_df_heuristics[b]
            #
            graph_edges = list(self.graph.edges)
            start_node, end_node = graph_edges[0][0], graph_edges[-1][1]
            print(start_node, end_node)
            # print(nx.astar_path(self.graph, source=start_node, target=end_node,
            #                     heuristic=heuristics_for_a_star, weight='cost'))
            break
        print(nx.is_tree(self.graph))
        return self

    def consistent_heuristic(self, parent_node, child_node, parent_to_child_cost):

        with open('jsons/heuristics.json', 'r') as file:
            heuristics_dict = json.load(file)
        heuristics_parent = heuristics_dict.get(parent_node)
        heuristics_child = heuristics_dict.get(child_node)
        heuristics_parent = min(heuristics_parent, parent_to_child_cost + heuristics_child)
        return heuristics_parent

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        # labels = nx.get_edge_attributes(self.graph, 'cost')
        # labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        # nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
        plt.show()


def store_graph(*args):
    """
    each arg is a list of tuples, where each tuple represents an edge between nodes.
    """
    # store starting edges from each path
    start_edges = sorted([tuple_ for path in args for (i, tuple_) in enumerate(path) if i == 0])
    #TODO if single listed is passed, no need to combine.

    # unpack the paths and chain them into a sorted list.
    path_combine = sorted(list(chain(*args)))
    # count the frequency of edges among start edges and total path edges
    start_edges_counter = Counter(start_edges)
    path_combine_counter = Counter(path_combine)
    # obtain distinc nodes from list of edge tuples
    node_list = sorted(set(list(chain.from_iterable(path_combine))))

    adjacency_df = pd.DataFrame(index=node_list, columns=node_list)
    adjacency_df.fillna(0, inplace=True)
    for key, value in path_combine_counter.items():
        adjacency_df.at[key[0], key[1]] = value
    return start_edges_counter, adjacency_df


def find_path(que, adjacency_matrix, average_path_length, path_list):
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
            que.append(successor_list[0])
            path_list.append(successor_list[0])
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
            max_node = max(look_ahead_df, key=lambda _key: look_ahead_df[_key])
            path_list.append(max_node[0])
            que.append(max_node[1])
            path_list.append(max_node[1])
    find_path(que, adjacency_matrix, average_path_length, path_list)
