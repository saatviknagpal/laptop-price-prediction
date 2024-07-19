## used for transforming data
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
import os

from utils import save_object

@dataclass
class DataTransformationConfig:
    file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''
    Handles data transformation processes
    '''
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def create_data_transformer(self):
        try:
            numerical_features = ["Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]

            categorical_features = [
                "Brand"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ])
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical features scaling completed")

            logging.info("Categorical features encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def start_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Loaded train and test data")

            preprocessing_obj = self.create_data_transformer()

            target_column = "Price"
            numerical_features = ["writing_score", "reading_score"]

            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_features_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_features_test_df = test_df[target_column]

            logging.info("Applying preprocessing to train and test data")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_features_train_df)
            ]
            test_arr = np.c_[
                input_features_test_arr, np.array(target_features_test_df)
            ]

            logging.info("Preprocessing object saved")

            save_object(
                file_path=self.data_transformation_config.file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)