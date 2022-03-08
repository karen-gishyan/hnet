import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict
from queue import PriorityQueue, Queue


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
    def start_node_to_node_i_heuristic(self):
        pass

    @abstractmethod
    def node_i_to_end_node_heuristic(self):
        pass

    @abstractmethod
    def between_node_heuristic(self):
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
    """

    def __init__(self, drugs_df: pd.DataFrame = None):
        self.graph = nx.Graph()
        self.drugs_df = drugs_df
        self.start_node = 'Emergency'
        self.end_node = 'Successful Discharge'
        # TODO: admittime dischtime probably need to be removed
        self.admittime = self.drugs_df.admittime[0]
        self.dischtime = self.drugs_df.dischtime[0]

    def start_node_to_node_i_heuristic(self):
        pass

    def node_i_to_end_node_heuristic(self, weight_dict):
        self.end_node = "Successful Discharge"
        node_edge_list = []
        for key in weight_dict['ELECTIVE'].keys():
            # lower the weight, higher the cost
            # TODO: weight logic may need to change depending on the model
            node_edge_list.append((self.end_node, key, {'cost': 1 - weight_dict['ELECTIVE'][key]}))
        return node_edge_list

    def between_node_heuristic(self) -> List[Tuple[str, str, Dict]]:
        """
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
         """
        total_overlap_list = []
        total_overlap = defaultdict(int)  # default is 0
        unique_admissions = self.drugs_df.hadm_id.unique()
        # TODO which of these in the end generate the route to the end_node
        # TODO direct node cannot exist between each node and end node

        for admission_i, id in enumerate(unique_admissions):
            patient_drugs = self.drugs_df[self.drugs_df.hadm_id == id]
            for i, row_i in patient_drugs.iterrows():
                for j, row_j in patient_drugs.iterrows():
                    if i == j:
                        continue
                    total_intersection_days = self.calculate_date_intersection(row_i, row_j)
                    if total_intersection_days:
                        patient_stay_length_days = (self.dischtime - self.admittime).days + 1
                        # cost = round(patient_stay_length_days / total_intersection_days, 2)
                        cost = round(total_intersection_days, 2)
                        # no need to store (i,j) if (j,i) already in keys
                        if not (row_j.drug, row_i.drug) in total_overlap.keys():
                            total_overlap[(row_i.drug, row_j.drug)] += total_overlap[(
                                row_i.drug, row_j.drug)] + cost

            print(f" Iteration {admission_i} / {len(unique_admissions)} done")
        for tuple_ in total_overlap:
            total_overlap_list.append((*tuple_, {'cost': total_overlap[tuple_]}))

        # TODO number of node-edges for first diagnosis (16373)
        # TODO inspect the very big cost numbers
        # TODO save results to json
        # return total_overlap_list
        self.graph.add_edges_from(total_overlap_list)
        edges_with_costs = self.graph.edges(data=True)
        return self

    def calculate_date_intersection(self, node_i, node_j) -> int:
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

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'cost')
        labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
        plt.show()
