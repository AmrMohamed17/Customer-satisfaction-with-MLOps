import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

def get_data_for_test():
  try:
    df = pd.read_csv("./data/olist_customers_dataset.csv")
    # required_columns = [
    #   'payment_sequential', 'payment_installments', 'payment_value',
    #   'price', 'freight_value', 'product_name_lenght',
    #   'product_description_lenght', 'product_photos_qty',
    #   'product_weight_g', 'product_length_cm',
    #   'product_height_cm', 'product_width_cm'
    # ]
    # df = df[required_columns]
    logging.info("Data loaded successfully.")

    df = df.sample(100)
    preprocess_strategy = DataPreprocessStrategy()
    data_cleaning = DataCleaning(df, preprocess_strategy)
    df = data_cleaning.handle_data()
    logging.info("Data cleaned successfully.")
    df.drop(["review_score"], axis=1, inplace=True)
    return df.to_json(orient="split")
  except Exception as e:
    logging.error(e)
    raise e
  