"""
Make the final prediction for the next X Months.
"""
from fire import Fire
from pickle import PicklingError

from src.utils import load_dataset, write_dataset, write_trained_model
from src.models import Models


def load_clean_data(project_path: str = "./"):
    """
    Load and the dataset to produce the final predictions.
    """
    data_path = project_path + "data/interim/"
    data = {}
    data["single_view"] = load_dataset("testset", data_path=data_path)
    data["transac"] = load_dataset("transac", data_path=data_path)
    return data


def make_prediction(estimator: str, horizon: int, project_path: str = "./"):
    """
    Make Prediction on the final dataset for the next months.
    """
    # Load data
    data = load_clean_data(project_path=project_path)
    # Load model
    model = Models().get(estimator)(discount_rate=0.3, horizon=horizon)
    # Fit & Predict
    model.fit(transac=data["transac"], single_view=data["single_view"])
    prediction = model.predict(data["single_view"])
    # Save Predictions
    write_dataset(prediction, "prediction", data_path=project_path + "data/output")
    # Save trained model
    try:
        write_trained_model(model, "model", model_path=project_path + "models/")
    except PicklingError:
        print("\nModel cannot be save due to pickling error.\n")


if __name__ == "__main__":
    Fire(make_prediction)
