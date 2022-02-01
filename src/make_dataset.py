"""
Make the analytical dataset use to train the prediction model.
"""
import os
from pathlib import Path

import pandas as pd
from fire import Fire
from src.utils import write_dataset

from utils import order_to_transac, clean_order, tag_new_repeat
from single_view import SingleView


def make_dataset(project_path: str = "./"):
    """
    Make the analytical dataset use to train the prediction model.
    """
    # Load data
    data_path = project_path + "data/raw/Online Retail.xlsx"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            """You must place the Online Retail dataset under the data/raw folder. """
            """\n                   Get the data at the link below: """
            """\n                   https://archive.ics.uci.edu/ml/datasets/Online+Retail"""
        )
    # Clean Up Order
    order = clean_order(order=pd.read_excel(data_path))
    # Order to Transac
    transac = order_to_transac(order)
    transac = tag_new_repeat(transac)
    transac_train = transac.query("timestamp <= '2011-08-31'")
    # Transac to Single View
    single_view_train = SingleView(transac_train).make()
    single_view = SingleView(transac).make()
    # Write datasets
    for dataset, name in zip(
        [single_view_train, single_view, transac, transac_train, order],
        ["trainset", "testset", "transac", "transactrain", "order"],
    ):
        write_dataset(
            dataset,
            dataset_name=name,
            data_path=Path(project_path + "data/interim"),
            verbose=True,
        )


if __name__ == "__main__":
    Fire(make_dataset)
