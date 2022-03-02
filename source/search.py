import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict

merged_df = pd.read_csv('./data/demo_merge.csv')
df = merged_df[
    ['subject_id', 'hadm_id_y', 'admittime', 'dischtime', 'admission_location', 'discharge_location', 'drug_type',
     'drug',
     'startdate', 'enddate', 'dose_val_rx', 'form_unit_disp', 'route', 'diagnosis']]
for column in ['startdate', 'enddate', 'admittime', 'dischtime']:
    df[column] = pd.to_datetime(df[column])
drugs = df.loc[:, ['admittime', 'hadm_id_y', 'dischtime', 'diagnosis', 'admission_location',
                   'discharge_location',
                   'drug', 'drug_type', 'startdate', 'enddate']]


# drugs = df.loc[:3, :]


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


class Graph(BaseGraph):
    """
    Construct a network, define edges.
    """

    def __init__(self, drugs_df: pd.DataFrame = None):
        self.graph = nx.Graph()
        self.drugs_df = drugs_df
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
        unique_admissions = self.drugs_df.hadm_id_y.unique()

        count = 0
        for id in unique_admissions:
            patient_drugs = self.drugs_df[self.drugs_df.hadm_id_y == id]
            for i, row_i in patient_drugs.iterrows():
                for j, row_j in patient_drugs.iterrows():
                    if i == j:
                        continue
                    total_intersection_days = self.calculate_date_intersection(row_i, row_j)
                    if total_intersection_days:
                        patient_stay_length_days = (self.dischtime - self.admittime).days + 1
                        cost = round(patient_stay_length_days / total_intersection_days, 2)
                        # no need to store (i,j) if (j,i) already in keys
                        if not (row_j.drug, row_i.drug) in total_overlap.keys():
                            total_overlap[(row_i.drug, row_j.drug)] += total_overlap[(
                                row_i.drug, row_j.drug)] + cost
            count += 1
            if count == 3:  # for three iterations, 2103 (node,node,weight) instances.
                break
        for tuple_ in total_overlap:
            total_overlap_list.append((*tuple_, {'cost': total_overlap[tuple_]}))

        return total_overlap_list

    def calculate_date_intersection(self, node_i, node_j) -> int:
        """
        Calculate the date of overlap between the serving of two drug nodes.
        The result is used for calculating the cost between two drugs.
        """
        format = '%m/%d/%Y %H:%M'
        # TODO format hour-minute is not correctly encoded, currently we do +1 on the total length.
        drug_i_range = DateTimeRange(node_i.startdate, node_i.enddate)
        drug_j_range = DateTimeRange(node_j.startdate, node_j.enddate)
        if drug_i_range.is_intersection(drug_j_range):
            intersection_date_range = drug_i_range.intersection(drug_j_range)
            intersections_total_days = max((intersection_date_range.end_datetime -
                                            intersection_date_range.start_datetime).days, 1)
            return intersections_total_days

    def search(self):
        pass

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'cost')
        labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
        plt.show()
