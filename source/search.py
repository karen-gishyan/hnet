import json
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from source.utils import calculate_date_intersection


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
