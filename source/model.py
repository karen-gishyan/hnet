from sklearn.svm import SVC
import pandas as pd
import numpy as np


def preprocess():
    """
    Preprocess. for model training
    :param data: 
    :return: df for model training.
    """
    admissions = pd.read_csv('data/ADMISSIONS.csv')
    prescriptions = pd.read_csv('data/PRESCRIPTIONS.csv', nrows=10000)

    unique_admissions = prescriptions.HADM_ID.unique()
    unique_drugs = np.sort(prescriptions.DRUG.unique())
    x = np.zeros(shape=(len(unique_admissions), len(unique_drugs)))
    admit_discharge_outcomes = np.empty(shape=(len(unique_admissions), 3), dtype='object')
    for adm_i, id in enumerate(unique_admissions):
        drug_array = np.zeros(shape=len(unique_drugs))  # for admission type, location, discharge location
        admission_drugs = prescriptions[prescriptions.HADM_ID == id]['DRUG'].to_numpy().flatten()
        admission_unique_drugs = np.sort(np.unique(admission_drugs))
        for drug_i, drug in enumerate(unique_drugs):
            if drug in admission_unique_drugs:
                drug_array[drug_i] = 1
        # assign the array to the specified row
        x[adm_i] = drug_array

        admission_type = admissions[admissions.HADM_ID == id]['ADMISSION_TYPE'].values[0]
        admission_location = admissions[admissions.HADM_ID == id]['ADMISSION_LOCATION'].values[0]
        discharge_location = admissions[admissions.HADM_ID == id]['DISCHARGE_LOCATION'].values[0]
        # store unique admission's admission, discharge details
        admit_discharge_outcomes[adm_i] = [admission_type, admission_location, discharge_location]

    x_df = pd.DataFrame(x, columns=unique_drugs)
    y_df = pd.DataFrame(admit_discharge_outcomes,
                        columns=['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION'])
    combined_df = pd.concat([x_df, y_df], axis=1)
    return combined_df


def svc(x, y):
    model = SVC(kernel='linear')
