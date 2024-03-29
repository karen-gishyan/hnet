from datetimerange import DateTimeRange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from textdistance import ratcliff_obershelp


def calculate_date_intersection(node_i, node_j) -> Optional[int]:
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


def diagnosis_value_counts(value_count=100) -> pd.Series:
    """
    Helper function
    Count the number of times each diagnosis appears in the admissions dataset, then
    Return series of drug names which appear at least value_counts number of times.

    :param int: value_count
    :return: pd. Series
    """
    df = pd.read_csv('data/ADMISSIONS.csv')
    unique_values = df.DIAGNOSIS.value_counts()
    unique_values = unique_values[unique_values >= value_count]
    return unique_values


def merge_dfs(drug_name=None) -> pd.DataFrame:
    """
    Merge admissions and prescriptions on either on a drug list or on a drug name.
    """
    admissions = pd.read_csv('data/ADMISSIONS.csv')
    prescriptions = pd.read_csv('data/PRESCRIPTIONS.csv')
    if drug_name:
        admissions = admissions[admissions.DIAGNOSIS == drug_name]
    df = pd.merge(admissions, prescriptions, on=['HADM_ID']).drop(['SUBJECT_ID_y', 'ROW_ID_y'], axis=1)
    df = df.rename(
        columns={'SUBJECT_ID_x': 'SUBJECT_ID', 'ROW_ID_x': 'ROW_ID'})
    df = df.rename(columns=str.lower)
    for column in ['startdate', 'enddate', 'admittime', 'dischtime']:
        df[column] = pd.to_datetime(df[column])
    return df


def unique_admission_per_diagnosis(merged_df):
    """
    Return pd.Series of diagnosis with their number of unique admissions.
    :param merged_df:
    :return:
    """
    n_unique_admissions_per_diagnosis = merged_df.groupby('diagnosis')['hadm_id'].nunique()
    sorted = n_unique_admissions_per_diagnosis.sort_values(ascending=False)
    sorted.to_csv('data/unique_admission_per_diagnosis.csv')
    return sorted


def evaluate(path, graph, number_of_admissions=3):
    """
    Return 'ratcliff_obershelp' score as a measure of sequence simillarity.
    :param path: resulting path of the algorithm
    :param graph: Graph instance
    :return:
    """
    path_string = ''.join(path)
    similarity_scores = []
    unique_admissions = number_of_admissions
    for adm_i, admission in enumerate(unique_admissions):
        patient_df = graph.df[(graph.df.hadm_id == admission)]
        patient_df.reset_index(inplace=True)
        drugs_string = ''.join(patient_df.drug)
        score = ratcliff_obershelp(path_string, drugs_string)
        similarity_scores.append(score)
        if (adm_i + 1) % 5 == 0:
            print(f'{adm_i + 1}/{len(number_of_admissions)} sequences evaluated.')
    return np.mean(similarity_scores), similarity_scores


def plot_results():
    """
    Visualize the results of the two algorithms for 10 diagnosis.
    """
    res_path_graph = pd.read_csv('data/results/path_graph_per_diagnosis_results.csv')
    res_binary_graph = pd.read_csv('data/results/binary_graph_per_diagnosis_results.csv')
    drug_names = pd.read_csv('data/unique_admission_per_diagnosis.csv')[:10]
    ypath = np.array(res_path_graph.values).flatten()
    ygraph = np.array(res_binary_graph.values).flatten()
    x = drug_names.diagnosis.values
    x[5] = 'CORONARY ARTERY BYPASS GRAFT/SDA'

    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    axs[0].bar(x, ypath)
    axs[0].set_xlabel('Diagnosis', labelpad=10)
    axs[0].set_ylabel('Score', labelpad=10)
    axs[0].set_title('Experiment 1 Results')
    axs[0].tick_params(axis='x', labelrotation=90)
    for i, j in zip(x, ypath):
        axs[0].annotate(str(round(j, 2)), xy=(i, j), ha='center', verticalalignment='bottom')

    axs[1].bar(x, ygraph)
    axs[1].set_xlabel('Diagnosis', labelpad=10)
    axs[1].set_ylabel('Score', labelpad=10)
    axs[1].set_title('Experiment 2 Results')
    axs[1].tick_params(axis='x', labelrotation=90)
    for i, j in zip(x, ygraph):
        axs[1].annotate(str(round(j, 2)), xy=(i, j), ha='center', verticalalignment='bottom')

    fig.suptitle('Mean Ratcliff-Obershelp score per diagnosis')
    plt.show()
