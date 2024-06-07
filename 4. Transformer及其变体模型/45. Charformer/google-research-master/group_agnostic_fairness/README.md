# Fairness without Demographics through Adversarially Reweighted Learning
This repository contains the code for the paper "Fairness without Demographics through Adversarially Reweighted Learning".

## Installation

Run the following:
```bash
virtualenv -p python3 .
source ./bin/activate

pip3 install -r group_agnostic_fairness/requirements.txt
```

## Data Preparation

The data provided in the './group_agnostic_fairness/data/toy_data directory is dummy, and is only for testing the code.
For meaningful results, please follow the steps below.

### Pre-process COMPAS dataset and create train and test files:

Download the COMPAS dataset from:
https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
and save it in the './group_agnostic_fairness/data/compas' folder.

Run './group_agnostic_fairness/data_utils/CreateCompasDatasetFiles.ipynb' notebook to process the dataset, and create files required for training.

### Pre-process UCI Adult (Census Income) dataset and create train and test files:

Download the Adult train and test data files can be downloaded from: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test and save them in the './group_agnostic_fairness/data/uci_adult' folder.

Run './group_agnostic_fairness/data_utils/CreateLawSchoolDatasetFiles.ipynb' notebook to process the dataset, and create files required for training.


### Pre-process Law School Admissions Council (LSAC) Dataset and create train and test files:

Download the Law School dataset from: (http://www.seaphe.org/databases.php), convert SAS file to CSV, and save it in the ./group_agnostic_fairness/data/law_school folder.

Run CreateLawSchoolDatasetFiles.ipynb to process the dataset, and create files required for training.


### Generate synthetic datasets used in the paper:

To generate various synthetic datasets used in the paper run './group_agnostic_fairness/data_utils/CreateUCISyntheticDataset.ipynb' notebook.

## Training and Inference

Training and evaluation for the adversarial reweighting model can be run as:

```bash
python -m group_agnostic_fairness/main_trainer
```
Refer to the test cases in <model_name>_model_test.py files to understand the workflow.

*Disclaimer: This is not an official Google product.*

## Hyper-parameter Settings

The following hyperparameters were used to generate the results reported in the paper:

| Dataset | Model |	Batch Size | Primary Learning Rate | Adversarial Learning Rate |
| ---- | ---- | ---- | ---- | ---- |
| UCI Adult |	Baseline |	256 |	0.01 |	NA |
| UCI Adult |	Adversarial Reweighting |	512 |	0.01 | 2 |
| UCI Adult |	IPW(S+Y) |	512 |	0.01 |	NA |
| UCI Adult |	IPW(S) |	64 |	0.001 |	NA |
| LAW SCHOOL |	Baseline |	128 |	0.1 |	NA |
| LAW SCHOOL | Adversarial Reweighting | 256 | 1 | 0.001 |
| LAW SCHOOL |	IPW(S+Y) |	512 |	0.1 |	NA |
| LAW SCHOOL |	IPW(S) |	512 |	0.1 |	NA |
| COMPAS |	Baseline |	512 |	0.01 |	NA |
| COMPAS |	Adversarial Reweighting |	512 |	0.01 |	0.001 |
| COMPAS |	IPW(S+Y) |	128 |	0.001 |	NA |
| COMPAS |	IPW(S) |	64 |	0.001 |	NA |

For the other hyperparameters including train steps, test steps, hidden units,
etc., the default values as specified in `main_trainer.py` were used.
