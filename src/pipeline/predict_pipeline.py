import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_data

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path="artifacts\preprocessor.pkl"
            model=load_data(file_path=model_path)
            preprocessor=load_data(file_path=preprocessor_path)
            transformed_features=preprocessor.transform(features)
            prediction=model.predict(transformed_features)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_html_as_df(self):
        try:
            data_input_html_dict={"gender":[self.gender],
                                  "race_ethinicity":[self.race_ethnicity],
                                  "parental_level_of_education":self.parental_level_of_education,
                                  "lunch":self.lunch,
                                  "test_preparation_course":self.test_preparation_course,
                                  "reading_score":self.reading_score,
                                  "writing_score":self.writing_score}
            return pd.DataFrame(data_input_html_dict)

        except Exception as e:
            raise CustomException(e,sys)    
        
