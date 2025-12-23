import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "random_forest_model.pkl")
    confusion_matrix_path = os.path.join("Notebook", "confusion_matrix.png")
    shap_summary_path = os.path.join("Notebook", "shap_summary.png")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting features and target from train and test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("Creating RandomForest Pipeline with SMOTE")
            model = ImbPipeline(steps=[
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    class_weight="balanced"
                ))
            ])

            logging.info("Training RandomForest Model")
            model.fit(X_train, y_train)

            logging.info("Making Predictions")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            logging.info(f"Test Accuracy: {acc}")
            print(f"Accuracy: {acc}")
            print("\nClassification Report:\n", classification_report(y_test, y_pred))

            # ------------------- Confusion Matrix -------------------
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            os.makedirs(os.path.dirname(self.config.confusion_matrix_path), exist_ok=True)
            plt.savefig(self.config.confusion_matrix_path)
            plt.close()
            logging.info(f"Confusion matrix saved at {self.config.confusion_matrix_path}")

            # ------------------- SHAP -------------------
            logging.info("Calculating SHAP values")
            rf = model.named_steps["clf"]
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_train)

            plt.figure()
            shap.summary_plot(shap_values, X_train, show=False)
            os.makedirs(os.path.dirname(self.config.shap_summary_path), exist_ok=True)
            plt.savefig(self.config.shap_summary_path, bbox_inches='tight')
            plt.close()
            logging.info(f"SHAP summary plot saved at {self.config.shap_summary_path}")

            # ------------------- Save Model -------------------
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)
            logging.info(f"RandomForest model saved at {self.config.trained_model_file_path}")

            return model, acc

        except Exception as e:
            logging.error("Exception occurred at Model Training")
            raise CustomException(e, sys)
