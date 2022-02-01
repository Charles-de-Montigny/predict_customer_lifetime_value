"""
Evaluation of the accuracy of the models.
"""
import click
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils import load_dataset, write_dataset
from src.models import Models


def load_clean_data(project_path: str = "./") -> dict:
    """
    Load and clean data for evaluation of the models.
    """
    data_path = project_path + "/data/interim"
    transac = load_dataset("transac_train", data_path=data_path)
    singleview = load_dataset("trainset", data_path=data_path)
    testset = load_dataset("testset", data_path=data_path)
    testset.rename(columns={"historic_clv": "actual_clv"}, inplace=True)
    testset = testset[["user_id", "actual_clv"]]
    return {"transac": transac, "singleview": singleview, "testset": testset}


def run_eval(estimator: str, verbose: bool = True, project_path: str = "./"):
    """
    Run the cross-validation of the model on the testset.

    Args:
        estimator: The name of the model from the Models object.
        verbose: Whether to print metrics to the console.
        project_path: 
    """
    # Load data
    data = load_clean_data(project_path=project_path)
    # Load model
    model = Models().get(estimator)(discount_rate=0.05, horizon=3)
    # Fit & Predict
    model.fit(transac=data["transac"], single_view=data["singleview"])
    prediction = model.predict(data["singleview"])
    prediction["predicted_clv"] = prediction["historic_clv"] + prediction["rlv"]
    prediction = pd.merge(prediction, data["testset"], on="user_id", how="left")
    # Compute metrics
    if verbose:
        click.secho("\nPREDICTION PRECISION:", bold=True)
        click.secho("----------------------", bold=True)
        print(
            "RMSE: ",
            np.round(
                np.sqrt(
                    mean_squared_error(
                        y_true=prediction["actual_clv"],
                        y_pred=prediction["predicted_clv"],
                    )
                ),
                1,
            ),
        )
        print(
            "MAE: ",
            np.round(
                mean_absolute_error(
                    y_true=prediction["actual_clv"], y_pred=prediction["predicted_clv"]
                ),
                1,
            ),
        )
        print(
            "Overprediction: ",
            np.round(
                np.sum(
                    (prediction["actual_clv"] < prediction["predicted_clv"])
                    / len(prediction)
                ),
                4,
            )
            * 100,
            " %",
        )
        click.secho("\n")
    # Save prediction
    write_dataset(
        prediction,
        data_path=project_path + "data/processed/",
        dataset_name="eval",
        verbose=verbose,
    )


if __name__ == "__main__":
    Fire(run_eval)
