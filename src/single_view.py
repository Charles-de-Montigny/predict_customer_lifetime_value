"""
Transforms a standard transactional dataframe into a single view one.
"""
import pathlib
from pathlib import Path

import click
import fire
import numpy as np
import pandas as pd

from utils import write_dataset

# from fluxion_utils.prediction.residual_lifetime import ResidualLifetime, SingleBuyerRLV
# from fluxion_utils.prediction.rfm import RFM, rfm_segment


class SingleView:
    """
    Transforms a standard transactional dataframe into a single view one.
    Standard columns are: 'user_id', 'total_price', 'timestamp', 'volume', 'item_id'
    """

    def __init__(
        self,
        transac: pd.DataFrame,
        project_path: Path = Path("./"),
        verbose: bool = False,
    ):
        """
        Constructor
        Args:
            input: input dataframe or input dataframe file name
        """
        self.project_path = project_path
        self.verbose = verbose

        # Setting the input dataframe, either directly or by loading a file
        self.transac = transac
        # Setting the columns name, please modify if columns don't match
        # self.col_names = load_column_config(project_path)
        # self.single_col_names = load_single_column_config(project_path)

        # Creating single view df with great length
        self.data = pd.DataFrame(self.transac["user_id"].unique(), columns=["user_id"])
        self.data.name = "singleview"

    def _add_columns(self, agg_func, name_dict=None, verbose=True):
        """
        Function to add a column to the single df based on a function and
        a name_dict

        Args:
            agg_func: Function that defines what will be in our column
            name_dict: Defines the input column and the new column in this format
                       {column_to_apply_function : new_column}
        """
        if self.verbose:
            click.secho(f"Add {agg_func.__name__}...")

        init_len = len(self.data)

        # Creates the new column to add to the single df,
        # either with or without a name_dict
        agg_df = agg_func()
        if name_dict:
            agg_df = pd.DataFrame(agg_df).rename(columns=name_dict)

        self.data = self.data.join(agg_df, on="user_id")

        final_len = len(self.data)
        if init_len != final_len:
            raise Exception("Lenght of the single view df must remain the same.")
        if verbose:
            click.secho(f"\n{agg_func.__name__}", bold=True)
            click.secho(f"Shape of Single View: {self.data.shape}", bold=True)

    def _add_frequency(self):
        """
        Add the number of visits per customer
        """

        def frequency():
            # Grouping by id and timestamp before grouping by id
            # this way we only count once transaction
            df_nb_purchases = self.transac.groupby(["user_id", "timestamp"])[
                "user_id"
            ].count()
            return df_nb_purchases.groupby(["user_id"]).count()

        name_dict = {"user_id": "frequency"}
        self._add_columns(frequency, name_dict)

    def _add_volume_sum(self):
        """
        Adds the sum of all bought volume
        """

        def volume_sum():
            return self.transac.groupby("user_id")["volume"].sum()

        name_dict = {"volume": "volume_sum"}
        self._add_columns(volume_sum, name_dict)

    def _add_most_recent_purchase(self):
        """
        Add the most recent purchase date
        """

        def most_recent_purchase():
            return pd.to_datetime(self.transac.groupby(["user_id"])["timestamp"].max())

        name_dict = {"timestamp": "most_recent_purchase"}
        self._add_columns(most_recent_purchase, name_dict)

    def _add_recency(self):
        def recency():
            most_recent_purchase = pd.to_datetime(
                self.transac.groupby(["user_id"])["timestamp"].max()
            )
            first_purchase = pd.to_datetime(
                self.transac.groupby(["user_id"])["timestamp"].min()
            )
            recency_df = (most_recent_purchase - first_purchase).to_frame()
            recency_df["timestamp"] = recency_df["timestamp"].apply(
                lambda x: x.total_seconds() / 3600 / 24
            )
            return recency_df

        name_dict = {"timestamp": "recency"}
        self._add_columns(recency, name_dict)

    def _add_first_purchase_date(self):
        """
        Add the first purchase date
        """

        def first_purchase():
            return pd.to_datetime(self.transac.groupby("user_id")["timestamp"].min())

        name_dict = {"timestamp": "first_purchase"}
        self._add_columns(first_purchase, name_dict)

    def _add_customer_age(self):
        """
        Add the age of the client: The time between max date and the first purchase.
        """

        def customer_age():
            max_date = self.transac["timestamp"].max()
            tmp_df = pd.to_datetime(max_date, utc=True) - self.transac.groupby(
                "user_id"
            )["timestamp"].min().to_frame().apply(lambda x: pd.to_datetime(x, utc=True))
            tmp_df["timestamp"] = tmp_df["timestamp"].apply(
                lambda x: x.total_seconds() / 24 / 3600
            )
            return tmp_df

        name_dict = {"timestamp": "customer_age"}
        self._add_columns(customer_age, name_dict)

    def _add_monetary(self):
        """
        Add the sum and the mean of price per customer
        """

        def monetary():
            return self.transac.groupby(["user_id"])["total_price"].agg(["mean", "sum"])

        name_dict = {
            "mean": "aov",
            "sum": "historic_clv",
        }
        self._add_columns(monetary, name_dict)

    def _add_days_between_purchases(self):
        """
        Add mean time between purchases. For a guest that purchased only once: np.nan
        """

        def days_between_purchases():
            self.transac["date"] = pd.to_datetime(self.transac["timestamp"])

            # New df sorted first by id then by date
            df = self.transac[["user_id", "date"]]
            df = df.sort_values(by=["user_id", "date"])

            # Creating two shifted df
            data0 = df.iloc[:-1, :].reset_index(drop=True)
            data0.rename(
                columns={"user_id": "id0", "date": "date0"},
                inplace=True,
            )
            data1 = df.iloc[1:, :].reset_index(drop=True)
            data1.rename(
                columns={"user_id": "id1", "date": "date1"},
                inplace=True,
            )

            # Calculating time between 2 purchases for each interval
            # between 2 visits and every customer
            data2 = pd.concat([data0, data1], axis=1)
            data2["days_between_purchases"] = np.where(
                data2["id0"] == data2["id1"],
                (data2["date1"] - data2["date0"]).apply(
                    lambda x: int(x.total_seconds() / 24 / 3600)
                ),
                np.nan,
            )

            # Time in days
            data2.set_index("id0", inplace=True)
            # Only when time between purchases is greater then a day
            data2 = data2.query("id0 == id1 and days_between_purchases > 1")

            # Return the mean
            return data2.groupby("id1")["days_between_purchases"].agg(["mean"])

        name_dict = {"mean": "mean_days_between_purchases"}
        self._add_columns(days_between_purchases, name_dict)

    def _add_value_segment(self):
        """
        Create segments based LTV (top 20% / 40% / 40%)
        """
        p4080 = self.data["historic_clv"].quantile(q=[0.4, 0.8])
        self.data["value_segment"] = np.where(
            self.data["historic_clv"] >= p4080.loc[0.8],
            "1 - high",
            np.where(
                (self.data["historic_clv"] >= p4080.loc[0.4])
                & (self.data["historic_clv"] < p4080.loc[0.8]),
                "2 - mid",
                "3 - low",
            ),
        )

    def _add_analytic(self) -> None:
        """
        Add analytics to the single view of customers.
        """
        self.data = ResidualLifetime().fit(self.data).transform()
        self.data = SingleBuyerRLV().fit(X=self.transac).transform(X=self.data)
        self.data = (
            RFM()
            .fit(self.data, relative_freq=False, use_active_prob=True)
            .transform(X=self.data)
        )
        self.data["rfm_segment"] = self.data.apply(
            lambda x: rfm_segment(recency=x["R"], freq=x["F"]), axis=1
        )
        if self.verbose:
            click.secho("Analytic: RLV, Active Prob & RFM", bold=True)
            click.secho(f"Shape of Single View: {self.data.shape}", bold=True)

    def make(self) -> None:
        """
        Function to make the transactional df a singleview df
        """
        if self.verbose:
            click.secho("\nMaking single view df...", bold=True)
        self._add_frequency()
        self._add_volume_sum()
        self._add_most_recent_purchase()
        self._add_first_purchase_date()
        self._add_recency()
        self._add_customer_age()
        self._add_monetary()
        self._add_days_between_purchases()
        self._add_value_segment()
        # self._add_analytic()
        return self.data

    def write(self, output_path=None, output_name="single"):
        """
        Writing the single df to parquet and csv
        Args:
            output_path: path were you want to write the file
            output_name: name of the file
        """
        output_path = (
            self.project_path / "data/processed"
            if output_path is None
            else pathlib.Path(output_path)
        )
        write_df(self.data, output_path, output_name, verbose=self.verbose)
