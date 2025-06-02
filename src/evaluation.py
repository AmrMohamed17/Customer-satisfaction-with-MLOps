import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
  @abstractmethod
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    pass


class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
      try:
        logging.info("Calculating MSE")
        mse =  mean_squared_error(y_pred=y_pred, y_true=y_true)
        logging.info("MSE: {}".format(mse))
        return mse
      except Exception as e:
        logging.error(e)
        raise 


class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
      try:
        logging.info("Calculating r2_score")
        r2 =  r2_score(y_pred=y_pred, y_true=y_true)
        logging.info("r2_score: {}".format(r2))
        return r2
      except Exception as e:
        logging.error(e)
        raise 


class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
      try:
        logging.info("Calculating RMSE")
        rmse =  np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
        logging.info("RMSE: {}".format(rmse))
        return rmse
      except Exception as e:
        logging.error(e)
        raise 
      
