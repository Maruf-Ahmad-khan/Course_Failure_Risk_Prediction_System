import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from category_encoders import TargetEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated for Suitability Dataset")

            # ------------------ Columns ------------------
            numerical_cols = ["Age"]

            cat_target_enc_cols = [
                "Gender",
                "City",
                "Highest_Qualification",
                "Stream",
                "Year_Of_Completion",
                "Are_you_currently_working",
                "Your_Designation",
                "Employment_Type"
            ]

            logging.info("Creating Encoding Pipelines")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )

            # Target Encoded Pipeline
            target_enc_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_enc", TargetEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("target_enc", target_enc_pipeline, cat_target_enc_cols),
                ],
                remainder="drop"
            )

            logging.info("ColumnTransformer Created Successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation Object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train & Test Data Loaded Successfully")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Suitability_Label"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessor on Train & Test")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df,
                target_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Transformation Completed")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor Saved Successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception Occurred in initiate_data_transformation")
            raise CustomException(e, sys)
