import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

merged_df = pd.read_csv('./data/demo_merge.csv')
df = merged_df[['subject_id', 'drug_type', 'drug', 'startdate', 'enddate', 'dose_val_rx', 'form_unit_disp', 'route']]
drugs = df.loc[df['subject_id'] == 10006, ['drug', 'drug_type', 'startdate', 'enddate']]
unique_drugs = len(drugs['drug'].unique())


# TODO: other search/graph trees, herusitic function, searching algorithm (modified or existing).
class MakeGraph:
    """
    Construct a network, define edges.
    """

    def __init__(self, drugs_df: pd.DataFrame):
        self.graph = nx.Graph()
        self.drugs_df = drugs_df
        self.df_length = len(self.drugs_df)

    def construct_nodes(self):
        self.graph.add_nodes_from(self.drugs_df.drug)

    def construct_combination_edges(self):
        """
         A Graph Tree
         Construct edges between drug nodes based on the fact if two drugs are used in combination.
         Combination means drugs are applied in the same timeframe.
        """
        self.construct_nodes()
        for i, row_i in self.drugs_df.iterrows():
            for j, row_j in self.drugs_df.iterrows():
                if i == j:
                    continue
                drug_i_start_date = datetime.strptime(row_i['startdate'], '%m/%d/%Y %H:%M')
                drug_i_end_date = datetime.strptime(row_i['enddate'], '%m/%d/%Y %H:%M')
                drug_j_start_date = datetime.strptime(row_j['startdate'], '%m/%d/%Y %H:%M')
                drug_j_end_date = datetime.strptime(row_j['enddate'], '%m/%d/%Y %H:%M')

                if (drug_j_start_date <= drug_i_start_date <= drug_j_end_date) or \
                        (drug_j_start_date <= drug_i_end_date <= drug_j_end_date):
                    self.graph.add_edge(row_i['drug'], row_j['drug'])
        # print(self.graph.edges)

        return self

    def construct_type_edges(self):
        self.construct_nodes()
        for i, row_i in self.drugs_df.iterrows():
            for j, row_j in self.drugs_df.iterrows():
                if i == j:
                    continue

                if row_i['drug_type'] == row_j['drug_type']:
                    self.graph.add_edge(row_i['drug'], row_j['drug'])
        return self

    def visualize(self):
        nx.draw(self.graph)
        plt.show()

    def __len__(self):
        return len(self.drugs_df)


if __name__ == "__main__":
    # MakeGraph(drugs).construct_edges().visualize()
    MakeGraph(drugs).construct_type_edges().visualize()
