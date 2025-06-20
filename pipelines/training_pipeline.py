from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evalute_model import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def train_pipeline(data_pth: str):
  df = ingest_df(data_pth)
  X_train, X_test, y_train, y_test = clean_df(df)
  model = train_model(X_train, X_test, y_train, y_test)
  r2, rmse = evaluate_model(model, X_test, y_test)
