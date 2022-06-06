from datetimerange import DateTimeRange
import pandas as pd
from typing import Optional


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


def merge_dfs(diagnosis_value_counts: pd.Series = None, drug_name=None,
              admission_df: pd.DataFrame = None, prescriptions_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge admissions and prescriptions on either on a drug list or on a drug name.
    """
    assert diagnosis_value_counts or drug_name, 'Diagnosis_value_counts or drug_name needs to be supplied'
    admissions = admission_df or pd.read_csv('data/ADMISSIONS.csv')
    prescriptions = prescriptions_df or pd.read_csv('data/PRESCRIPTIONS.csv')
    if diagnosis_value_counts:
        admissions = admissions[admissions.DIAGNOSIS.isin(diagnosis_value_counts.index)]
    else:
        admissions = admissions[admissions.DIAGNOSIS == drug_name]
    df = pd.merge(admissions, prescriptions, on=['HADM_ID']).drop(['SUBJECT_ID_y', 'ROW_ID_y'], axis=1)
    df = df.rename(
        columns={'SUBJECT_ID_x': 'SUBJECT_ID', 'ROW_ID_x': 'ROW_ID'})
    df = df.rename(columns=str.lower)
    for column in ['startdate', 'enddate', 'admittime', 'dischtime']:
        df[column] = pd.to_datetime(df[column])
    return df
