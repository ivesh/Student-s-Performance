import sys,os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class Datatransformationconfig:
    """Data Transformation Configuration Class"""
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class Datatransformation:
    """Data Transformation Class"""
    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()

    def get_data_transformer_obj(self):
        """Get Transformer Object is used to convert categorical to numerical and process other transformation"""
        try:
            numerical_columns=['reading_score','writing_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline= Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("one_hot_encoder",OneHotEncoder()),("scaler",StandardScaler(with_mean=False))])
            logging.info("Numerical Columns scaling done")
            logging.info("Categorical Columns encoding done")
            processesor=ColumnTransformer([("num_pipepline",num_pipeline,numerical_columns),("cat_pipeline",cat_pipeline,categorical_columns)])
            return processesor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Training dat set read")
            logging.info("Testing dat set read")
            preprocessing_obj=self.get_data_transformer_obj()
            target_column="math_score"
            input_features_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_features_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            print()
            print(input_features_test_df)
            logging.info(f"Applying preprocessing techniques to transform train and test data.")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)
            train_arr= np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saved preprocessed data")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)    


