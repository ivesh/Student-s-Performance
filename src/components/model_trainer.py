import os,sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            logging.info("Splitted training data into X-train,y-train,X-test,y-test, Now reading models")
            models={
                "Linear Regression":LinearRegression(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "KNN Regressor":KNeighborsRegressor(),
                "XGBoost Regressor":XGBRFRegressor()                
            }
            logging.info("Models are read, Now training models")
            model_report:dict= evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            logging.info("Model evaluation is done, Now fetching the score report for the best model")
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            logging.info("Best model is being selected, Now checking model performance")
            if best_model_score < 0.6:
                raise CustomException("Model accuracy is less than 0.6")
            logging.info("Best model is selected, Now saving it")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info("Model is saved")
            predicted=best_model.predict(X_test)
            r2_sc=r2_score(y_test, predicted)
            return r2_sc 
        
        except Exception as e:
            raise CustomException(e,sys)