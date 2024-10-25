import os,sys 
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import Datatransformation,Datatransformationconfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class Dataingestionconfig:
    train_path:str=os.path.join('artifacts',"train.csv")
    test_path:str=os.path.join('artifacts',"test.csv")
    raw_data:str=os.path.join('artifacts',"data.csv")

class Dataingestion:
    def __init__(self):
        self.ingestion_config=Dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("The ingestion method is running inside the data_ingestion.py file which is in components")
        try:
            #here data can be read from mongodb and other sources
            df=pd.read_csv("src/notebook/data/student.csv")
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data, index=False,header=True)
            logging.info("Data ingestion is completed")
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
            logging.info("Data ingestion is completed")
            return(self.ingestion_config.train_path,self.ingestion_config.test_path) 


        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=Dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()   
    #Combining data_transformation
    data_transform=Datatransformation()
    train_arr,test_arr,_=data_transform.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))
                 