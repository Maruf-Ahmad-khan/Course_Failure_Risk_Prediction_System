from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


if __name__ == "__main__":
    logging.info("====== Machine Learning Pipeline Started ======")

    try:
        # 1️⃣ Data Ingestion
        ingestion_obj = DataIngestion()
        train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

        # 2️⃣ Data Transformation
        transform_obj = DataTransformation()
        train_arr, test_arr, _ = transform_obj.initiate_data_transformation(
            train_path=train_data_path, 
            test_path=test_data_path
        )

        # 3️⃣ Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)

        logging.info("====== Pipeline Execution Successful ======")

    except Exception as e:
        logging.error("Pipeline Failed")
        raise CustomException(e, sys)
