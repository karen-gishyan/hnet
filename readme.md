## hnet
Repository of path construction algorithms based on patient
admission and drug prescriptions using clinical datasets.

To replicate the experiments, you need access to **ADMISSIONS.csv** and 
**PRESCRIPTIONS.csv** files from **MIMIC-III** Clinical Database,
which need to be present in the **data** subdirectory inside **hnet** directory.
You then need to provide pd.DataFrame objects of these files
to **merge_df()** function inside the **main** module.
