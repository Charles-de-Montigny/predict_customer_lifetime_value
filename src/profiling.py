from datetime import datetime as dt
from pathlib import Path

import pandas as pd
from pandas_profiling import ProfileReport
from fire import Fire

from src.utils import load_dataset


def profiling(
    df: pd.DataFrame, project_path: Path = Path("./"), minimal: bool = False
) -> None:
    """
    Creates a pandas profiling report in HTML.
    """
    today = dt.now().strftime("%Y-%m-%d")
    profile = ProfileReport(
        df, title="Profiling Report: Chess Match-up", minimal=minimal
    )
    profile.to_file(project_path.joinpath(f"reports/profiling_{today}.html"))
    return


def main(dataset, minimal) -> None:
    data = load_dataset(dataset, data_path="data/interim", return_X_y=False)
    profiling(data, minimal=minimal)


if __name__ == "__main__":
    Fire(main)
