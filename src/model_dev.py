import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
  @abstractmethod
  def train(self, X_train, y_train):
    pass


class LRModel(Model):
  def train(self, X_train, y_train, **kwargs):
    try:
      lr = LinearRegression()
      lr.fit(X_train, y_train)
      logging.info("Model training Completed")
      return lr
    except Exception as e:
      logging.error(e)
      raise e