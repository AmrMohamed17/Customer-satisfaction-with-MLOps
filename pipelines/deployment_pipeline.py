from io import StringIO
import json
import logging
import numpy as np
import pandas as pd
import requests
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from steps.config import ModelNameConfig
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.steps import Output
from pydantic import BaseModel
from pipelines.utils import get_data_for_test

from steps.clean_data import clean_df
from steps.ingest_data import ingest_df
from steps.evalute_model import evaluate_model
from steps.train_model import train_model  

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@step
def deployment_trigger(
    accuracy: float,
    min_accuracy: float = 0.92,
):
    return accuracy > min_accuracy

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Load the prediction service."""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if existing_services:
        service = existing_services[0]
        # if running and not service.is_running:
        #     service.start()
        # elif not running and service.is_running:
        #     service.stop()
        return service
    else:
        raise ValueError("No active deployment found.")
    

# @step
# def predictor(
#     service: MLFlowDeploymentService,
#     data: str,
# ) -> np.ndarray:
#     service.start(timeout=10)
#     data = json.loads(data)
#     # logging.info(f"Data shape: {np.array(data['data']).shape}")
#     # logging.info(data)
#     data.pop("columns")
#     data.pop("index")
#     columns_for_df = [
#         "payment_sequential",
#         "payment_installments",
#         "payment_value",
#         "price",
#         "freight_value",
#         "product_name_lenght",
#         "product_description_lenght",
#         "product_photos_qty",
#         "product_weight_g",
#         "product_length_cm",
#         "product_height_cm",
#         "product_width_cm",
#     ]
#     logging.info(f"Shape of data['data']: {np.array(data['data']).shape}")
#     df = pd.DataFrame(data["data"], columns=columns_for_df)
#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#     data = np.array(json_list)
#     payload = {
#         "columns": columns_for_df,
#         "data": data.values.tolist(),
#     }
#     predictions = service.predict(data)
#     return predictions

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)
    
    # Parse data
    df = pd.read_json(StringIO(data), orient='split')
    data_array = df.values
    
    logging.info(f"Input data shape: {data_array.shape}")
    logging.info(f"Service running: {service.is_running}")
    
    # Get the prediction URL from the service
    try:
        if hasattr(service.endpoint, 'prediction_url'):
            url = service.endpoint.prediction_url
            logging.info(f"Using prediction_url: {url}")
        else:
            # Extract URL components from the service endpoint
            host = getattr(service.endpoint, 'ip_address', '127.0.0.1')
            
            # Try to get port from different possible locations
            port = None
            if hasattr(service.endpoint, 'status') and hasattr(service.endpoint.status, 'port'):
                port = service.endpoint.status.port
            elif hasattr(service.endpoint, 'port'):
                port = service.endpoint.port
            else:
                port = 8000  # Default MLFlow port
            
            url = f"http://{host}:{port}/invocations"
            logging.info(f"Constructed URL: {url}")
            
    except Exception as url_error:
        logging.error(f"Failed to get service URL: {url_error}")
        # Log service endpoint details for debugging
        logging.info(f"Service endpoint type: {type(service.endpoint)}")
        logging.info(f"Service endpoint attributes: {dir(service.endpoint)}")
        if hasattr(service.endpoint, 'status'):
            logging.info(f"Endpoint status attributes: {dir(service.endpoint.status)}")
        raise url_error
    
    # Prepare payload in MLFlow dataframe_split format
    payload = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": data_array.tolist()
        }
    }
    
    try:
        logging.info(f"Sending prediction request to: {url}")
        logging.info(f"Payload keys: {list(payload.keys())}")
        logging.info(f"Number of samples: {len(payload['dataframe_split']['data'])}")
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            predictions = response.json()
            logging.info(f"Prediction successful: {len(predictions)} predictions returned")
            # logging.info(f"First few predictions: {predictions[:3] if len(predictions) > 3 else predictions}")
            return np.array(predictions)
        else:
            logging.error(f"Prediction failed with status {response.status_code}")
            logging.error(f"Response headers: {dict(response.headers)}")
            logging.error(f"Response content: {response.text}")
            
            # Try alternative formats
            alternative_payloads = [
                {"instances": data_array.tolist()},
                {"inputs": data_array.tolist()},
                {"data": data_array.tolist()}
            ]
            
            for i, alt_payload in enumerate(alternative_payloads):
                try:
                    logging.info(f"Trying alternative format {i+1}: {list(alt_payload.keys())}")
                    alt_response = requests.post(url, json=alt_payload, timeout=60)
                    
                    if alt_response.status_code == 200:
                        predictions = alt_response.json()
                        logging.info(f"Alternative format {i+1} successful!")
                        return np.array(predictions)
                    else:
                        logging.warning(f"Alternative format {i+1} failed: {alt_response.status_code}")
                        
                except Exception as alt_error:
                    logging.warning(f"Alternative format {i+1} error: {alt_error}")
            
            # If all alternatives fail, raise the original error
            raise Exception(f"All prediction formats failed. Status: {response.status_code}, Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        
        # Try to use the service.predict method as fallback
        try:
            logging.info("Trying service.predict method as fallback")
            predictions = service.predict(data_array)
            logging.info(f"Service.predict fallback successful")
            return predictions
        except Exception as service_error:
            logging.error(f"Service.predict fallback also failed: {service_error}")
            raise Exception(f"Both direct HTTP and service.predict failed. HTTP error: {e}, Service error: {service_error}")
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise e

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)
    deploy_decision = deployment_trigger(r2, min_accuracy)
    mlflow_model_deployer_step(
            deploy_decision=deploy_decision,
            model=model,
            workers=workers,
            timeout=timeout,
        )
    # deployer()


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
    )
    data = dynamic_importer()
    predictions = predictor(service=service, data=data)
    return predictions
