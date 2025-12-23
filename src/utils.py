import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
     try:
          dir_path = os.path.dirname(file_path)
          
          os.makedirs(dir_path, exist_ok=True)
          
          with open(file_path, "wb") as file_obj:
               pickle.dump(obj, file_obj)
               
     except Exception as e:
          raise CustomException(e, sys)
     


def evaluate_model(X_train, y_train, X_test, y_test, models, plot_confusion=True):
    """
    Trains and evaluates classification models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: dict of models {"model_name": model_instance}
        plot_confusion: bool, whether to plot confusion matrix

    Returns:
        report: dict of model_name -> accuracy
        class_reports: dict of model_name -> classification report dict
        conf_matrices: dict of model_name -> confusion matrix
    """
    try:
        report = {}
        class_reports = {}
        conf_matrices = {}

        for model_name, model in models.items():
            logging.info(f"Training Model: {model_name}")

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_test_pred = model.predict(X_test)

            # Accuracy
            test_accuracy = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_accuracy

            # Classification Report
            class_report_dict = classification_report(y_test, y_test_pred, output_dict=True)
            class_reports[model_name] = class_report_dict

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            conf_matrices[model_name] = cm

            # Logging
            logging.info(f"\nAccuracy for {model_name}: {test_accuracy:.4f}")
            logging.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_test_pred)}")

            # Optional Confusion Matrix Plot
            if plot_confusion:
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix - {model_name}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.show()

        return report, class_reports, conf_matrices

    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)

               
         
     
     
def load_object(file_path):
     try:
          with open(file_path, 'rb') as file_obj:
               return pickle.load(file_obj)
               
               
     except Exception as e:
          logging.info("Exception occured in load_object funtion util")
          raise CustomException(e, sys)