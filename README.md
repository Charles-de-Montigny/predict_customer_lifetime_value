Predict Customer Lifetime Value
==============================

The goal is to create a method to predict the lifetime value of customers.

Installation
------------

You need to have Python installed on your machine.

MacOS & Linux:
```
git clone https://github.com/Charles-de-Montigny/predict_customer_lifetime_value.git
cd predict_customer_lifetime_value
bash ./init.sh
```

Getting Started
----------------

Once the installation is completed, proceed as follows to produce a result file:

1- Get the data:

Download the data from this [link](https://archive.ics.uci.edu/ml/datasets/Online+Retail) and unzip the data in the data/raw folder.

```
data/raw/Online Retail.xlsx
```

2- From the project root, run the bash script:
```
cd "to project root"
bash ./run_prediction.sh
```

3- Retrieve the result file in data/processed/:
```
data/output/prediction_{YYYY-MM-DD}.csv
```

Profiling Reports
-----------------

There is a pandas profiling report available in the reports folder but if you want to run it yourself:

```
python src/profiling.py --dataset train
python src/profiling.py --dataset test
```

For more information about the pandas Profiling project:
[pandas profiling documentation](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/)


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── run_prediction.sh    <- A standalone script that runs the whole pipeline on the verification set and creates a prediction file.
    │
    ├── config             <- Configuration files for colors and columns to used in analysis.
    │
    ├── data
    │   ├── raw            <- The original, immutable data dump.
    │   ├── interim        <- Interim steps in data preparation.
    │   ├── processed      <- Data sets for modeling.
    │   └── output         <- Test tournament with predictions.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML and Notebooks.
    │   ├── metrics        <- Data sets for modeling.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
       ├── __init__.py     <- Makes src a Python module.
       │
       ├── make_dataset.py <- Scripts to download or generate data.
       │
       ├── run_eval.py     <- Scripts to evaluate models performance.
       │
       ├── make_prediction.py     <- Scripts to produce final predictions.
       │
       └── profiling.py  <- Scripts to create exploratory and results oriented visualizations.


--------
