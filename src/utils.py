"""
Utility functions for this project.
"""
import joblib
import json
import re
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def days_between_purchases(
    df: pd.DataFrame, timestamp: str = "timestamp", user_id: str = "user_id"
) -> pd.DataFrame:
    """
    Compute days between two purchases for a given customer.

    Args:
        df: Dataframe with transactions.
        timestamp: Date columns in the DataFrame.
        user_id: The ID of the customer in the dataframe.

    Returns a pandas DataFrame with of the mean between purchase for every customers.
    """
    df["date"] = pd.to_datetime(df[timestamp])
    # New df sorted first by id then by date
    df = df[[user_id, "date"]]
    df = df.sort_values(by=[user_id, "date"])
    # Creating two shifted df
    data0 = df.iloc[:-1, :].reset_index(drop=True)
    data0.rename(columns={user_id: "id0", "date": "date0"}, inplace=True)
    data1 = df.iloc[1:, :].reset_index(drop=True)
    data1.rename(columns={user_id: "id1", "date": "date1"}, inplace=True)
    # Calculating time between 2 purchases for each interval
    # between 2 visits and every customer
    data2 = pd.concat([data0, data1], axis=1)
    data2["days_between_purchases"] = np.where(
        data2["id0"] == data2["id1"],
        (data2["date1"] - data2["date0"]).apply(
            lambda x: x.total_seconds() / 24 / 3600
        ),
        np.nan,
    )
    # Time in days
    data2.set_index("id0", inplace=True)
    # Only when time between purchases is greater then a day
    data2 = data2.query("id0 == id1 and days_between_purchases > 1")
    # Return the mean
    return data2.groupby("id1")["days_between_purchases"].agg(["mean"])


def tag_new_repeat(
    transac: pd.DataFrame,
    user_id: str = "user_id",
    timestamp: str = "timestamp",
    new_col_name: str = "new",
):
    """
    Goes through a transactional dataframe and tag the line where a customer is
    considered new

    Args:
        transac: the transactional dataframe
        user_id: the name of the user id column in the dataframe
        timestamp: the name of the timestamp column in the dataframe
        new_col_name: name of the column containing if the client is new
    Returns: A copy of the dataframe with a new column indicating the
             transaction at wich the customer is considered new
    """
    df = transac.copy(deep=True)
    df = df.sort_values(by=timestamp)

    df["cumcount"] = df.groupby(user_id).cumcount()
    df[new_col_name] = df["cumcount"] == 0
    return df


def order_to_transac(
    order: pd.DataFrame,
    user_id: str = "user_id",
    timestamp: str = "timestamp",
    aggregation: dict = {"total_price": "sum", "volume": "sum"},
):
    """
    Transforms an order dataframes with multiple items of the same
    transaction on multiple lines to a true transac dataframe with all
    infos of a transaction in the same line

    Args:
        order: The order dataframe we want to transform
        user_id: Name of the column containing the id of the entity
                 making the transaction. Default value assumes df is in
                 a standard transactional dataframe
        timestamp: name of column for the timestamp the transaction was made at
                   Default value assumes df is in the standarddataframe format
        aggregation: the way columns in commun should be aggregated,
                     default value assumes that column names are the ones of
                     a standard transactional dataframe
    Returns: The transactional dataframe with all infos of a transaction
             on the same line
    """
    order["timestamp"] = pd.to_datetime(order["timestamp"]).apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    transac = order.groupby([user_id, timestamp]).agg(aggregation).reset_index()
    transac.name = "transac"
    return transac.sort_values(by=timestamp)


def read_json(input_path):
    with open(input_path) as file:
        data = json.load(file)
    return data


def get_max_date(
    name: str,
    data_path: Path,
    extension: str = ".parquet",
    date_format: str = "%Y-%m-%d",
    s3_bucket=None,
) -> str:
    """
    Date of the most recent interim file starting with string.
    Args:
        name: start of the name of the file.
        data_path: Path to the folder.
        extension: The extension to look for.
        date_format: The format of datetime in file name.
        s3_bucket: The S3 bucket object from the boto3 library.
    Returns: date string of the most recent interim transac file
    """
    if date_format not in ["%Y-%m-%d", "%Y-%m-%d-%H%M%S"]:
        raise ValueError("Datetime format is not allowed.")
    if date_format == "%Y-%m-%d":
        max_date = base_date = dt.strptime("1900-01-01", date_format)
    elif date_format == "%Y-%m-%d-%H%M%S":
        max_date = base_date = dt.strptime("1900-01-01-000000", date_format)
    if s3_bucket:
        files = []
        for file in s3_bucket.objects.all():
            files.append(file.key)
    else:
        files = os.listdir(data_path)
    for file in files:
        date = re.findall(
            f"^{name}_+(.*)\{extension}", file
        )  # pylint: disable=anomalous-backslash-in-string
        if date:
            date = dt.strptime(date[0], date_format)
            max_date = date if date > max_date else max_date
    if max_date == base_date:
        raise FileNotFoundError("No corresponding file....")
    max_date = max_date.strftime(date_format)
    return max_date


def get_X_y(df: pd.DataFrame, target: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Get the form X, y from a dataframe.
    """
    y = df[target].values
    X = df.drop(target, axis=1)
    return X, y


def load_dataset(
    dataset_name: str = "trainset",
    data_path: Path = Path("data/interim"),
    return_X_y: bool = False,
    extension: str = ".parquet",
    target: str = "target",
    sample: float = None,
    date: str = None,
    s3_bucket=None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset for modeling.
    Args:
        dataset_name: The name of the dataset.
        data_path: The path to the data folder.
        return_X_y: Whether to return X, y or a unique dataframe.
        extension: The extension of the file.
        target: The column with the target variable.
        sample: [0, 1], if None no sample is computed.
        get_max_date: Whether to find the max date associate with the name.
        verbose: If True, print the name of the loaded file.
    Returns the dataset in either X, y or df.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if extension not in [".csv", ".parquet", ".json"]:
        raise ValueError(
            f"Extension must be either .csv, .json, .xlsx or .parquet but is {extension} ..."
        )
    if sample is not None and (sample > 1 or sample < 0):
        raise ValueError(f"Sample must be between 0 and 1 but is {sample}.")
    if date is None:
        date = "_" + get_max_date(
            dataset_name, data_path, extension=extension, s3_bucket=s3_bucket
        )
    else:
        date = "_" + date
    interim_data_path = data_path.joinpath("".join([dataset_name, date, extension]))
    if verbose:
        print(f"\nThe file {interim_data_path} is loaded...")
    if s3_bucket:
        obj = s3_bucket.Object("".join([dataset_name, date, extension])).get()
        df = read_json(obj["Body"], is_s3=True)
    else:
        if extension == ".parquet":
            df = pd.read_parquet(interim_data_path)
        elif extension == ".csv":
            df = pd.read_csv(interim_data_path)
        elif extension == ".json":
            df = read_json(interim_data_path)
        elif extension == ".xlsx":
            df = pd.read_excel(interim_data_path)
    if sample:
        df = df.sample(frac=sample)
    if return_X_y:
        if target not in df.columns:
            raise KeyError(
                f"The target variable {target} is not in the loaded dataset..."
            )
        return get_X_y(df, target=target)
    return df


def write_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    data_path: Path = Path("data/raw"),
    extension: str = ".parquet",
    verbose: bool = True,
) -> None:
    """
    Write raw data.
    Args:
        df: The dataset to write as a file.
        dataset_name: The name of the dataset.
        data_path: The path to the data folder.
        extension: The extension of the file.
        verbose: If True, print the name of the loaded file.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if dataset_name[-1] != "_":
        dataset_name += "_"
    if extension not in [".csv", ".parquet", ".json"]:
        raise ValueError(
            f"Extension must be .csv, .parquet or .json but is {extension}."
        )
    today = dt.now().strftime("%Y-%m-%d")
    output_path = data_path.joinpath("".join([dataset_name, today, extension]))
    if extension == ".parquet":
        df.to_parquet(output_path, index=False)
    elif extension == ".csv":
        df.to_csv(output_path, index=False)
    elif extension == ".json":
        with open(output_path, "w") as file:
            json.dump(df, file)
    if verbose:
        print(f"Dataset has been save to: {output_path}")


def write_trained_model(
    model, model_name: str, project: str = None, model_path: Path = Path("models")
) -> None:
    """
    Write trained model to a pickle file.
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    today = dt.now().strftime("%Y-%m-%d")
    if project:
        model_project_path = model_path.joinpath(project)
        Path.mkdir(model_project_path, exist_ok=True)
        full_path = model_project_path.joinpath(f"{model_name}_{today}.pkl")
    else:
        full_path = model_path.joinpath(f"{model_name}_{today}.pkl")
    joblib.dump(model, full_path)


def clean_order(order: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Orders: renaming and fixing types.
    """
    order.rename(
        columns={
            "InvoiceNo": "invoice_number",
            "StockCode": "stock_code",
            "Description": "product",
            "Quantity": "volume",
            "UnitPrice": "total_price",
            "CustomerID": "user_id",
            "InvoiceDate": "timestamp",
            "Country": "country",
        },
        inplace=True,
    )
    order.dropna(subset=["user_id"], inplace=True)
    order["user_id"] = order["user_id"].astype(int)
    order["stock_code"] = order["stock_code"].astype(str)
    order["invoice_number"] = order["invoice_number"].astype(str)
    return order
