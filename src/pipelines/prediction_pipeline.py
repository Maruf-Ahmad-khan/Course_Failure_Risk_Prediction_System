import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        try:
            # Load preprocessor and trained RandomForest model
            self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            self.model_path = os.path.join('artifacts', 'random_forest_model.pkl')

            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)

            logging.info("Preprocessor and model loaded successfully for prediction")
        except Exception as e:
            logging.error("Error initializing PredictPipeline")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Transform features using the saved preprocessor
            features_transformed = self.preprocessor.transform(features)
            
            # Predict using the trained RandomForest model
            predictions = self.model.predict(features_transformed)
            return predictions
        except Exception as e:
            logging.error("Exception occurred during prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: float,
                 Gender: str,
                 City: str,
                 Highest_Qualification: str,
                 Stream: str,
                 Year_Of_Completion: int,
                 Are_you_currently_working: str,
                 Your_Designation: str,
                 Employment_Type: str,
                 First_Name: str = None,
                 Last_Name: str = None,
                 Company_Name: str = None):
        self.Age = Age
        self.Gender = Gender
        self.City = City
        self.Highest_Qualification = Highest_Qualification
        self.Stream = Stream
        self.Year_Of_Completion = Year_Of_Completion
        self.Are_you_currently_working = Are_you_currently_working
        self.Your_Designation = Your_Designation
        self.Employment_Type = Employment_Type
        self.First_Name = First_Name
        self.Last_Name = Last_Name
        self.Company_Name = Company_Name

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "City": [self.City],
                "Highest_Qualification": [self.Highest_Qualification],
                "Stream": [self.Stream],
                "Year_Of_Completion": [self.Year_Of_Completion],
                "Are_you_currently_working": [self.Are_you_currently_working],
                "Your_Designation": [self.Your_Designation],
                "Employment_Type": [self.Employment_Type],
                "First_Name": [self.First_Name],
                "Last_Name": [self.Last_Name],
                "Company_Name": [self.Company_Name],
            }

            df = pd.DataFrame(data_dict)
            logging.info("CustomData converted to DataFrame")
            return df
        except Exception as e:
            logging.error("Exception occurred while creating DataFrame from CustomData")
            raise CustomException(e, sys)
