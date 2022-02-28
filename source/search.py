import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from datetimerange import DateTimeRange

merged_df = pd.read_csv('./data/demo_merge.csv')
df = merged_df[['subject_id', 'admittime', 'dischtime', 'admission_location', 'discharge_location', 'drug_type', 'drug',
                'startdate', 'enddate', 'dose_val_rx', 'form_unit_disp', 'route', 'diagnosis']]
for column in ['startdate', 'enddate', 'admittime', 'dischtime']:
    df[column] = pd.to_datetime(df[column])
drugs = df.loc[
    df['subject_id'] == 10006, ['admittime', 'dischtime', 'diagnosis', 'admission_location', 'discharge_location',
                                'drug', 'drug_type', 'startdate', 'enddate']]
drugs = df.loc[:10, :]


# TODO: other search/graph trees, herusitic function, searching algorithm (modified or existing).
class MakeGraph:
    """
    Construct a network, define edges.
    """

    def __init__(self, drugs_df: pd.DataFrame, directed=False):
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.drugs_df = drugs_df
        self.df_length = len(self.drugs_df)
        self.start_node = self.drugs_df.admission_location[0]
        self.end_node = self.drugs_df.discharge_location[0]
        self.admittime = self.drugs_df.admittime[0]
        self.dischtime = self.drugs_df.dischtime[0]
        self.diagnosis = self.drugs_df.diagnosis[0]

    def calculate_date_overlap(self, node_i, node_j):
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
        return

    def construct_combination_edges(self):
        """
         A Graph Tree
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
        """
        for i, row_i in self.drugs_df.iterrows():
            # from the starting node and end_node to the resulting drugs the cost is 0, we define costs
            # between the drugs.
            self.graph.add_edges_from(
                [(self.start_node, row_i.drug, {'cost': 0}), (self.end_node, row_i.drug, {'cost': 0})])
            for j, row_j in self.drugs_df.iterrows():
                if i == j:
                    continue
                # TODO: 1,10
                if self.graph.has_edge(row_i.drug, row_j.drug):
                    continue
                print(i, j)  # TODO:  1, 10 and 10, 1 both exist, inspect.
                total_intersection_days = self.calculate_date_overlap(row_i, row_j)
                if total_intersection_days:
                    patient_stay_length_days = (self.dischtime - self.admittime).days + 1
                    cost = round(patient_stay_length_days / total_intersection_days, 2)
                    self.graph.add_edge(row_i.drug, row_j.drug, cost=cost)

        return self

    def construct_hetero_combination_edges(self):
        """
        Start and final stated added plus nodes are connected at a chronological order.
        """
        self.construct_hetero_nodes()
        unique_dates = self.drugs_df.startdate.unique()
        for i, date in enumerate(unique_dates):
            drugs_based_on_date = self.drugs_df[self.drugs_df.startdate == date].drug
            if not i == len(unique_dates) - 1:  # stop connection for the last date
                drugs_based_on_next_date = self.drugs_df[self.drugs_df.startdate == unique_dates[i + 1]].drug
            for drug in drugs_based_on_date:
                if i == 0:
                    self.graph.add_edges_from([(self.start_node, drug)])
                else:
                    self.graph.add_edges_from([(drug, i) for i in drugs_based_on_next_date])

    def construct_type_edges(self):
        """
        Connect drugs if of the same type.
        """
        self.construct_nodes()
        for i, row_i in self.drugs_df.iterrows():
            for j, row_j in self.drugs_df.iterrows():
                if i == j:
                    continue
                if row_i.drug_type == row_j.drug_type:
                    self.graph.add_edge(row_i.drug, row_j.drug)
        return self

    def visualize(self):
        position = nx.spring_layout(self.graph)
        nx.draw(self.graph, position, with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'cost')
        labels = {edge: self.graph.edges[edge]['cost'] for edge in self.graph.edges}
        nx.draw_networkx_edge_labels(self.graph, position, edge_labels=labels)
        plt.show()

    def __len__(self):
        return len(self.drugs_df)


if __name__ == "__main__":
    MakeGraph(drugs).construct_combination_edges().visualize()
