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
            params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNN Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11, 13],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 4, 5, 6, 7, 8]
                }
                
            }
            logging.info("Models are read, Now training models")
            model_report:dict= evaluate_model(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,models=models,params=params)
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