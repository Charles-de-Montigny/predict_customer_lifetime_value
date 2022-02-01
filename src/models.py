"""
Model use to predict customer lifetime value.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, ParetoNBDFitter, GammaGammaFitter
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

from utils import days_between_purchases


class SingleBuyer(BaseEstimator):
    """
    Estimate the probability and the amount of a second purchase for a unique purchase customer.
    """

    def __init__(self, model: BaseEstimator = KNeighborsRegressor) -> None:
        self.model = model
        self.is_fitted = False

    @staticmethod
    def _get_repeat_customers(transac: pd.DataFrame) -> pd.DataFrame:
        """Only keep the customers of that bought multiples times."""
        repeat_customers_list = (
            transac.groupby("user_id")["cumcount"]
            .max()
            .to_frame()
            .query("cumcount > 0")
            .index.to_list()
        )
        transac["is_repeat"] = transac["user_id"].apply(
            lambda x: x in repeat_customers_list
        )
        repeat_transac = transac.loc[(transac["is_repeat"]) & (transac["cumcount"] < 2)]
        return repeat_transac

    @staticmethod
    def _compute_time_between_purchase(repeat_transac: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time between the first and the second purchase.
        """
        repeat_time = days_between_purchases(repeat_transac)
        repeat_time.reset_index(drop=False, inplace=True)
        repeat_time.rename(columns={"mean": "days_until_repeat"}, inplace=True)
        return repeat_time

    @staticmethod
    def _compute_active_probabibilty(repeat_time: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the probabibilty of being active (repurchase) for a single buyer client.
        """
        quantile = [i for i in np.arange(0, 1, 0.01)]
        active_prob = (
            repeat_time["days_until_repeat"]
            .quantile(q=quantile)
            .to_frame()
            .reset_index()
            .rename({"index": "active_prob"}, axis=1)
        )
        active_prob["active_prob"] = 1 - active_prob["active_prob"]
        return active_prob

    @staticmethod
    def _get_X_y(active_prob: pd.DataFrame) -> Tuple[np.array]:
        """
        Transform the active prob dataframe in X, y.
        """
        X = active_prob["days_until_repeat"].to_numpy().reshape(-1, 1)
        y = active_prob["active_prob"].to_numpy()
        return X, y

    def fit(self, transac: pd.DataFrame) -> SingleBuyer:
        """
        Fit model.
        """
        # Data cleaning
        repeat_transac = self._get_repeat_customers(transac)
        repeat_time = self._compute_time_between_purchase(repeat_transac)
        active_prob = self._compute_active_probabibilty(repeat_time)
        X, y = self._get_X_y(active_prob)
        self.fitted_model = self.model().fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, single_view: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the repurchase probability and the future CLV.
        """
        check_is_fitted(self, "is_fitted")
        single_buyers = single_view.query("frequency == 1")
        # Repeat odds
        repeat_odds = np.mean(single_view["frequency"] > 1)
        # Compute probability of repurchase
        predicted_probability = self.fitted_model.predict(
            X=single_buyers["customer_age"].to_numpy().reshape(-1, 1)
        )
        single_buyers["active_prob"] = predicted_probability * repeat_odds
        single_buyers["rlv"] = single_buyers["active_prob"] * single_buyers["aov"]
        return single_buyers


class MultiBuyer(BaseEstimator):
    """
    Estimate the probability and the amount of futures purchases for a repeat customer.
    """

    transac_models = {"bg": BetaGeoFitter, "nbd": ParetoNBDFitter}

    def __init__(
        self,
        horizon: int = 12,
        beta_penalizer: float = 0.0,
        gg_penalizer: float = 0.0,
        discount_rate: float = 0.01,
        transaction_prediction_model: str = "bg",
    ) -> None:
        self.horizon = horizon
        self.discount_rate = discount_rate
        self.transaction_prediction_model = self.transac_models[
            transaction_prediction_model
        ](penalizer_coef=beta_penalizer)
        self.gg_model = GammaGammaFitter(penalizer_coef=gg_penalizer)
        self.repeat_customers: pd.DataFrame = None
        self.is_fitted: bool = False

    def fit(self, single_view: pd.DataFrame) -> MultiBuyer:
        """
        Fit the prediction model.
        """
        self.input_args = {
            "frequency": single_view["frequency"] - 1,
            "recency": single_view["recency"],
            "T": single_view["customer_age"],
        }
        self.repeat_customers = single_view.query("frequency > 1")
        # Transaction prediction model
        self.transaction_prediction_model.fit(**self.input_args)
        # Gamma Gammg Model for CLV prediction
        self.gg_model.fit(
            frequency=self.repeat_customers["frequency"] - 1,
            monetary_value=self.repeat_customers["aov"],
            time=self.horizon,
        )
        self.is_fitted = True
        return self

    def predict(self) -> pd.DataFrame:
        """
        Predict the repurchase probability and the future CLV.
        """
        check_is_fitted(self, "is_fitted")
        # Predict Probability of Being <<Alive>>
        self.repeat_customers[
            "active_prob"
        ] = self.transaction_prediction_model.conditional_probability_alive(
            frequency=self.repeat_customers["frequency"] - 1,
            recency=self.repeat_customers["recency"],
            T=self.repeat_customers["customer_age"],
        )
        # Predict the future CLV
        self.repeat_customers["rlv"] = self.gg_model.customer_lifetime_value(
            transaction_prediction_model=self.transaction_prediction_model,
            frequency=self.repeat_customers["frequency"] - 1,
            recency=self.repeat_customers["recency"],
            T=self.repeat_customers["customer_age"],
            monetary_value=self.repeat_customers["aov"],
            time=self.horizon,
            discount_rate=self.discount_rate,
        )
        return self.repeat_customers


class ResidualLifetimeValue(BaseEstimator):
    def __init__(
        self,
        horizon: int = 365,
        beta_penalizer: float = 0.0,
        gg_penalizer: float = 0.0,
        discount_rate: float = 0.01,
        transaction_prediction_model: str = "bg",
    ) -> None:
        self.single_buyer = SingleBuyer()
        self.multi_buyer = MultiBuyer(
            horizon,
            beta_penalizer,
            gg_penalizer,
            discount_rate,
            transaction_prediction_model,
        )

    def fit(self, transac: pd.DataFrame, single_view: pd.DataFrame):
        self.single_buyer.fit(transac)
        self.multi_buyer.fit(single_view=single_view)
        return self

    def predict(self, single_view: pd.DataFrame) -> pd.DataFrame:
        single_buyer_sv = self.single_buyer.predict(single_view=single_view)
        multi_buyer_sv = self.multi_buyer.predict()
        single_view_w_prediction = pd.concat([single_buyer_sv, multi_buyer_sv], axis=0)
        return single_view_w_prediction


class ModelNotFoundError(Exception):
    """
    Error for when the model is missing from the data.
    """

    def __init__(self, message="Model not found."):
        super().__init__(message)


class Models:
    """
    Model's object.
    """

    probabilistic = ResidualLifetimeValue

    def get(self, estimator: str) -> BaseEstimator:
        """
        Get a model.
        """
        model = getattr(self, estimator)
        if not model:
            raise ModelNotFoundError()
        return model
