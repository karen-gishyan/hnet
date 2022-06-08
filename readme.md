## hnet
Repository of path construction algorithms based on patient
admission and drug prescriptions using clinical datasets.

To replicate the experiments, you need access to **ADMISSIONS.csv** and 
**PRESCRIPTIONS.csv** files from **MIMIC-III** Clinical Database,
which need to be present in the **data** subdirectory inside **hnet** directory.

## Running the project 
### virtual environment
1. `cd` into the `hnet` directory
2. On Linux/MacOS:
    1. `python3 -m venv venv`
    2. `source venv/bin/activate`
    3. `pip3 install -r requirements.txt`
    4. `python3 source/main.py`
3.  On Windows:
    1. `python -m venv venv`
    2. `venv\Scripts\activate.bat`
    3. `pip install -r requirements.txt`
    4. `python source/main.py`

### Docker    
1. `make build` or `docker build -t hnet .`
2. `make run-hnet` or `docker-compose up -d && docker exec -it hnet python3 source/main.py`