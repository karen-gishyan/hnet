from datetimerange import DateTimeRange
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from typing import Dict, Optional


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
    Count the number of times each diagnosis appears in the admissions dataset, then
    Return series of drug names which appear at least value_counts number of times.

    :param int: value_count
    :return: pd. Series
    """
    df = pd.read_csv('../data/ADMISSIONS.csv')
    unique_values = df.DIAGNOSIS.value_counts()
    unique_values = unique_values[unique_values >= value_count]
    return unique_values


def merge_dfs(diagnosis_value_counts: pd.Series = None, drug_name=None) -> pd.DataFrame:
    """
    Merge admissions and prescriptions on either on a drug list or on a drug name.
    """
    assert diagnosis_value_counts or drug_name, 'Diagnosis_value_counts or drug_name needs to be supplied'
    admissions = pd.read_csv('../data/ADMISSIONS.csv')
    prescriptions = pd.read_csv('../data/PRESCRIPTIONS.csv')
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


def make_train_data(df) -> pd.DataFrame:
    """
    Constructs a dataframe where the drugs are the features X and the final patient discharge is the
    outcome y. Each row represents a unique hospital admittance. From the total list of drugs,
    if the drug was given to a patient during his recovery period, a drug is assigned a value of 1,
    otherwise 0.

    :return: dataframe of drugs as features for each patient to be used for model training.
    """
    unique_admissions = df.hadm_id.unique()
    unique_drugs = np.sort(df.drug.unique())
    X = np.zeros(shape=(len(unique_admissions), len(unique_drugs)))
    admit_discharge_outcomes = np.empty(shape=(len(unique_admissions), 1), dtype='object')
    for adm_i, id in enumerate(unique_admissions):
        drug_array = np.zeros(shape=len(unique_drugs))  # for admission type, location, discharge location
        admission_drugs = df[df.hadm_id == id]['drug'].to_numpy().flatten()
        admission_unique_drugs = np.sort(np.unique(admission_drugs))
        for drug_i, drug in enumerate(unique_drugs):
            if drug in admission_unique_drugs:
                drug_array[drug_i] = 1
        # assign the array to the specified row
        X[adm_i] = drug_array
        # admission_type = df[df.hadm_id == id]['admission_type'].values[0]
        discharge_location = df[df.hadm_id == id]['discharge_location'].values[0]
        # store unique admission's admission, discharge details
        admit_discharge_outcomes[adm_i] = [discharge_location]

    X_df = pd.DataFrame(X, columns=unique_drugs)
    Y_df = pd.DataFrame(admit_discharge_outcomes,
                        columns=['discharge_location'])
    # If a patient is discharged alive is assigned a label of 1
    Y_df['discharge'] = Y_df.discharge_location.apply(lambda x: 0 if x == "DEAD/EXPIRED" else 1)
    Y_df.drop(['discharge_location'], axis=1, inplace=True)
    combined_df = pd.concat([X_df, Y_df], axis=1)
    return combined_df


def get_drug_weights(df) -> Dict:
    """
    :param df: combined df of x and y, the result of the make_train_data() function.
    :return: dictionary of weights for each drug as a result of a model training.
    """
    weights = dict()
    y = df.discharge
    X = df.drop(['discharge'], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    feature_to_weight = dict(zip(X.columns, model.coef_))
    weights.update({type: feature_to_weight})
    return weights
