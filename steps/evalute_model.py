import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, RMSE, R2
import mlflow
from zenml.client import Client

experiment = Client().active_stack.experiment_tracker



@step(experiment_tracker= experiment.name)
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame,y_test: pd.DataFrame,) -> Tuple[Annotated[float, "r2"], Annotated[float, "rmse"]]:
  try:
    y_pred = model.predict(X_test)
    mse = MSE().calculate_scores(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    r2 =  R2().calculate_scores(y_test, y_pred)
    mlflow.log_metric("r2", r2)

    rmse = RMSE().calculate_scores(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)
    
    return r2, rmse
  except Exception as e:
    logging.error(e)
    raise