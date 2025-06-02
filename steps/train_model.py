import logging
import mlflow.sklearn
import pandas as pd
from zenml import step
from zenml.client import Client
from src.model_dev import LRModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow

experiment_track = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_track.name)
def train_model(
  X_train: pd.DataFrame,
  X_test: pd.DataFrame,
  y_train: pd.DataFrame,
  y_test: pd.DataFrame,
)-> RegressorMixin:
  model = None
  config=ModelNameConfig()
  if config.model_name == "LinearRegression":
    mlflow.sklearn.autolog()
    # print("on it.")
    trained_model = LRModel().train(X_train, y_train)
    return trained_model
  else:
    raise ValueError("Model {} not listed.".format(config.model_name))