from datetimerange import DateTimeRange
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from typing import Dict, Optional
import json


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
    Select
    :param value_count:
    :return:
    """
    df = pd.read_csv('data/ADMISSIONS.csv')
    # shows how often each diagnosis appears.
    unique_values = df.DIAGNOSIS.value_counts()
    # TODO >= is correct, others are for experimentation
    unique_values = unique_values[unique_values <= value_count]  # at least 100 cases for each diagnosis type
    return unique_values


def merge_admissions_prescriptions(diagnosis_value_counts: pd.Series) -> pd.DataFrame:
    """
    Merge dfs by first selecting admissions where each diagnosis appears at least n number of times (default n=100)
    """
    admissions = pd.read_csv('data/ADMISSIONS.csv')
    prescriptions = pd.read_csv('data/PRESCRIPTIONS.csv')
    admissions = admissions[admissions.DIAGNOSIS.isin(diagnosis_value_counts.index)]
    df = pd.merge(admissions, prescriptions, on=['HADM_ID']).drop(['SUBJECT_ID_y', 'ROW_ID_y'], axis=1)
    df = df.rename(
        columns={'SUBJECT_ID_x': 'SUBJECT_ID', 'ROW_ID_x': 'ROW_ID'})
    df = df.rename(columns=str.lower)
    for column in ['startdate', 'enddate', 'admittime', 'dischtime']:
        df[column] = pd.to_datetime(df[column])
    return df

def make_train_data() -> pd.DataFrame:
    """
    Constructs a dataframe where the drugs are the features X and the final patient discharge is the
    outcome y. Each row represents a unique hospital admittance. From the total list of drugs,
    if the drug was given to a patient during his recovery period, a drug is assigned a value of 1,
    otherwise 0.
    :return: dataframe of drugs as features for each patient to be used for model training.
    """
    # 15692 unique diagnosis
    admissions = pd.read_csv('data/ADMISSIONS.csv')
    prescriptions = pd.read_csv('data/PRESCRIPTIONS.csv', nrows=10000)

    unique_admissions = prescriptions.HADM_ID.unique()
    unique_drugs = np.sort(prescriptions.DRUG.unique())
    X = np.zeros(shape=(len(unique_admissions), len(unique_drugs)))
    admit_discharge_outcomes = np.empty(shape=(len(unique_admissions), 2), dtype='object')
    for adm_i, id in enumerate(unique_admissions):
        drug_array = np.zeros(shape=len(unique_drugs))  # for admission type, location, discharge location
        admission_drugs = prescriptions[prescriptions.HADM_ID == id]['DRUG'].to_numpy().flatten()
        admission_unique_drugs = np.sort(np.unique(admission_drugs))
        for drug_i, drug in enumerate(unique_drugs):
            if drug in admission_unique_drugs:
                drug_array[drug_i] = 1
        # assign the array to the specified row
        X[adm_i] = drug_array

        admission_type = admissions[admissions.HADM_ID == id]['ADMISSION_TYPE'].values[0]
        discharge_location = admissions[admissions.HADM_ID == id]['DISCHARGE_LOCATION'].values[0]
        # store unique admission's admission, discharge details
        admit_discharge_outcomes[adm_i] = [admission_type, discharge_location]

    X_df = pd.DataFrame(X, columns=unique_drugs)
    Y_df = pd.DataFrame(admit_discharge_outcomes,
                        columns=['ADMISSION_TYPE', 'DISCHARGE_LOCATION'])
    # If a patient is discharged alive is assigned a label of 1
    Y_df['DISCHARGE'] = Y_df.DISCHARGE_LOCATION.apply(lambda x: 0 if x == "DEAD/EXPIRED" else 1)
    Y_df.drop(['DISCHARGE_LOCATION'], axis=1, inplace=True)
    combined_df = pd.concat([X_df, Y_df], axis=1)
    return combined_df


def svc(df) -> Dict:
    """
    :param df: combined df of x and y, the result of the model_preprocess() function.
    :return: dictionary of weights for each drug as a result of a model training.
    """
    weights = dict()
    unique_admission_type = df.ADMISSION_TYPE.unique()
    # TODO: iterating over the admission types may be needed to be moved during instantiation
    for type in unique_admission_type:
        X = df[df.ADMISSION_TYPE == type]
        y = X.DISCHARGE
        X = X.drop(['ADMISSION_TYPE', 'DISCHARGE'], axis=1)
        model = LinearRegression()
        model.fit(X, y)
        feature_to_weight = dict(zip(X.columns, model.coef_))
        weights.update({type: feature_to_weight})
    return weights
