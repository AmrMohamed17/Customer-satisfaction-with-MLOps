import logging
import pandas as pd
from zenml import step

class IngestData:
  def __init__(self, data_path: str):
    self.data_path = data_path

  def get_data(self):
    logging.info(f"Ingesting data from {self.data_path}")
    # cols = pd.read_csv(self.data_path)
    # print(f"columnsaskf: {cols.columns}")
    return pd.read_csv(self.data_path)
  
@step
def ingest_df(data_path: str) -> pd.DataFrame:
  try:
    return IngestData(data_path).get_data()
  except Exception as e:
    logging.error(f"error while ingesting data: {e}")
    raise e
  