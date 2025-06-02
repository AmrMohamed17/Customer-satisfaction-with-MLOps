import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Datastrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(Datastrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Encode categorical (object) columns using LabelEncoder
            label_encoders = {}
            for col in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[col] = data[col].fillna("missing") 
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(data[col].mean())

            return data
        except Exception as e:
            logging.error("Preprocessing failed", exc_info=True)
            raise e


class DataDivideStrategy(Datastrategy):
    def handle_data(self, data: pd.DataFrame):
        try:
            if 'review_score' not in data.columns:
                raise ValueError("'review_score' column not found in dataset.")

            y = data['review_score']
            X = data.drop(columns=['review_score'])

            # Ensure X and y have the same number of rows
            assert len(X) == len(y), f"Inconsistent samples: X={len(X)}, y={len(y)}"

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Data splitting failed", exc_info=True)
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: Datastrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Strategy failed", exc_info=True)
            raise e
