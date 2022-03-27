import json
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from networkx.algorithms.traversal import dfs_tree, dfs_successors, edge_dfs, dfs_edges
from networkx import astar_path
from source.utils import svc, model_preprocess


# TODO: working A_star,limiting edges between nodes, and limit search space based on some diagnosis grouping.

class BaseGraph(ABC):
    """
    Parent class for the Graph class with abstract methods to be implemented.
    The graph is not homogenous, having the following structure:
    start_node: str, "admission"
    goal_node: str , "discharge"
    all other nodes: str, "drug"

    Logic from start node to any a given node_i, a between node heuristic and from a given node_i to goal_node
    need to be implemented.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_node_to_node_i_cost(self):
        pass

    @abstractmethod
    def node_i_to_end_node_cost(self):
        pass

    @abstractmethod
    def between_node_cost(self):
        pass

    def A_Star(self, start_node, end_node):
        """
        https://mat.uab.cat/~alseda/MasterOpt/AStar-Algorithm.pdf
        https://www.baeldung.com/cs/a-star-algorithm
        """
        assert self.graph() is not None, "nx.Graph() does not exist."
        self.explored = set()
        self.frontier = set()
        self.g_n_dict = {}
        self.h_n_dict = {}  # the heuristics from each node need to exist
        self.parent_dict = {}
        self.g_n_dict.update({"Emergency": 0})
        self.h_n_dict.update({"Emergency": 100})  # TODO: check for admissability
        self.frontier.add(start_node)
        while self.frontier:
            current_node = self.node_with_cheapest_f(self.frontier)
            if current_node is end_node:
                break
            successor_nodes = self.generate_successor(current_node)
            for successor in successor_nodes:
                successor_cost = self.g(current_node) + self.cost_between_nodes(current_node, successor)
                if successor in self.frontier:  # if in frontier then self.g() is known
                    if self.g(successor) <= successor_cost:  # if existing cost is smaller than newly generated_cost
                        continue
                elif successor in self.explored:
                    if self.g(successor) <= successor_cost:
                        self.explored.remove(successor)
                        self.frontier.add(successor)
                        continue
                else:
                    self.frontier.add(successor)
                    # TODO does h(node_successor) need to be set or it is fixed?
                self.g_n_dict[successor] = successor_cost
                self.parent_dict.update({current_node: successor})
            self.explored.add(current_node)

        if current_node != end_node:
            raise ValueError('Solution has not been found but no more nodes in the frontier')

    def node_with_cheapest_f(self, frontier) -> str:
        """
        Return the node from the frontier with the lowest f(n)=g(n)+h(n).
        :return: cost
        """
        fronter_fs = dict()
        for i, node in enumerate(frontier):
            fronter_fs.update({node: self.f(node)})
        min_node_and_f = min(fronter_fs.items(), key=lambda x: x[1])
        return min_node_and_f[0]

    def generate_successor(self, node) -> List[str]:
        """
        generate list of successor nodes from a given node.
        """
        successor_list = []
        for (u, v) in self.graph.edges():
            if node == u:
                successor_list.append(v)
            elif node == v:
                successor_list.append(u)
            else:
                raise KeyError(f'node {node} is not a graph node.')
        return successor_list

    def cost_between_nodes(self, current_node, successor_node):
        for (u, v, cost) in self.graph.edges.data('cost'):
            if (u == current_node and v == successor_node) or (v == current_node and u == successor_node):
                return cost

    def g(self, current_node):
        return self.g_n_dict.get('current_node')

    def h(self, current_node):
        return self.h_n_dict.get('current_node')

    def f(self, current_node):
        return self.g(current_node) + self.h(current_node)


class Graph(BaseGraph):
    """
    Construct a network, define edges.
    Tests:
    1_admission: ✓
    """

    if not os.path.isdir('jsons'): os.mkdir('jsons')

    def __init__(self, diagnosis_df: pd.DataFrame = None):
        self.graph = nx.Graph()
        self.df = diagnosis_df
        self.unique_admissions = diagnosis_df.hadm_id.unique()
        self.diagnosis_name = self.df.diagnosis[0]
        super(Graph, self).__init__()

    def start_end_to_drug_cost(self):
        """
        Rationale, see how often a given drug is given on the first day of admittance, and on the last day of
        discharge, this become the respective cost between start_node_to_drug and drug_to_end_node.
        """
        drug_weight_start = defaultdict(int)
        drug_weight_end = defaultdict(int)
        for admission in self.unique_admissions:
            patient_df = self.df[self.df.hadm_id == admission]
            admittime = patient_df.admittime[0]
            dischtime = patient_df.dischtime[0]
            for i, row in patient_df.iterrows():
                # even if does not exist, create and add one
                # TODO adjust the cost before updating
                if row.startdate.date() == admittime.date():
                    # key has to be str for json encoding
                    drug_weight_start['{}'.format(('start_node', row.drug))] += 1
                if row.enddate.date() == dischtime.date():
                    drug_weight_end['{}'.format(('end_node', row.drug))] += 1

        with open(f"jsons/'{self.diagnosis_name}'_drug_weight_start.json", "w") as file:
            json.dump(drug_weight_start, file)

        with open(f"jsons/'{self.diagnosis_name}'_drug_weight_end.json", "w") as file:
            json.dump(drug_weight_end, file)

    def start_node_to_node_i_cost(self):
        pass

    def node_i_to_end_node_cost(self):
        pass

    def between_node_cost(self):
        """
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
         Tests:
         1_admission: ✓
         """
        total_overlap = defaultdict(int)  # default is 0
        for adm_i, admission in enumerate(self.unique_admissions):
            patient_df = self.df[self.df.hadm_id == admission]
            drug_cost_dict = {}
            for i, row_i in patient_df.iterrows():
                for j, row_j in patient_df.iterrows():
                    if i == j:
                        continue
                    total_intersection_days = self.calculate_date_intersection(row_i, row_j)
                    if total_intersection_days:
                        # TODO dischtime, admittime needs to be checked.
                        cost = patient_stay_length_days = (row_i.dischtime - row_i.admittime).days + 1
                        # cost = round(patient_stay_length_days / total_intersection_days, 2)
                        # no need to store (i,j) if (j,i) already in keys
                        if not '{}'.format((row_j.drug, row_i.drug)) in total_overlap.keys():
                            total_overlap['{}'.format((row_i.drug, row_j.drug))] += cost
            break

            print(f" Iteration {adm_i} / {len(self.unique_admissions)} done")

        with open(f"jsons/{self.diagnosis_name}_drugs.json", "w") as file:
            json.dump(total_overlap, file)

    def calculate_date_intersection(self, node_i, node_j) -> Optional[int]:
        """
        Calculate the date of overlap between the serving of two drug nodes.
        The result is used for calculating the cost between two drugs.
        """
        format = '%m/%d/%Y %H:%M'
        # TODO format hour-minute is not correctly encoded, currently we do +1 on the total length.
        # checks for erroneous data startdate is bigger than enddate
        if node_i.startdate > node_i.enddate or node_j.startdate > node_j.enddate:
            return
        drug_i_range = DateTimeRange(node_i.startdate, node_i.enddate)
        drug_j_range = DateTimeRange(node_j.startdate, node_j.enddate)
        if drug_i_range.is_intersection(drug_j_range):
            intersection_date_range = drug_i_range.intersection(drug_j_range)
            intersections_total_days = max((intersection_date_range.end_datetime -
                                            intersection_date_range.start_datetime).days, 1)
            return intersections_total_days

    def calculate_heuristic(self):
        model_weights = svc(
            model_preprocess())  # svc should be renamed to model_weights, allowing for different models.
        with open(f"jsons/{self.diagnosis_name}_drug_heuristics.json", "w") as file:
            json.dump(model_weights, file)

    def return_heuristic(self, node_i, node_j):
        """
        Return heuristic for the given node to end_node.
        """
        with open(f"jsons/{self.diagnosis_name}_drug_heuristics.json", "r") as file:
            heuristics = json.load(file)
        return heuristics.keys['(node_i,node_j)']

    def load_files(self):

        with open(f"jsons/{self.diagnosis_name}_drug_weight_start.json", "r") as file:
            start_to_drugs = json.load(file)
        with open(f"jsons/{self.diagnosis_name}_drugs.json", "r") as file:
            between_drugs = json.dump(file)
        with open(f"jsons/{self.diagnosis_name}_drug_weight_end.json", "r") as file:
            drugs_to_end = json.dump(file)

        return start_to_drugs, between_drugs, drugs_to_end

    def construct_graph(self):
        """
        Convert dicts ojff nodes and edges to nx.add_edges_from() format, then construct the graph.
        """
        start_to_drugs, between_drugs, drugs_to_end = self.load_files()

        start_to_drug_edges, between_drugs_edges, drugs_to_end_edges = [], [], []
        for tuple_ in start_to_drugs:
            start_to_drug_edges.append((*tuple_, {'cost': start_to_drugs[tuple_]}))
        for tuple_ in start_to_drugs:
            between_drugs_edges.append((*tuple_, {'cost': between_drugs[tuple_]}))
        for tuple_ in start_to_drugs:
            drugs_to_end_edges.append((*tuple_, {'cost': drugs_to_end[tuple_]}))

        # add the list elements together and construct the final graph.
        start_to_drug_edges.extend(between_drugs_edges)
        start_to_drug_edges.extend(drugs_to_end_edges)
        self.graph.add_edges_from(start_to_drug_edges)

    def nx_a_star(self):
        print(nx.astar_path(self.graph, 'start_node', 'end_node', heuristic=self.return_heuristic, weight='cost'))

    def bfs(self):
        edges = self.graph.edges(data=True)

        res1 = sorted(list(dfs_tree(self.graph, source='NS', depth_limit=50).edges()))
        res2 = dfs_successors(self.graph)
        # res = sorted(list(dfs_tree(self.graph, source='NS')))
        res3 = list(edge_dfs(self.graph, source='NS'))
        res4 = list(dfs_edges(self.graph, source='NS'))
        return self

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        # labels = nx.get_edge_attributes(self.graph, 'cost')
        # labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        # nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
        plt.show()
