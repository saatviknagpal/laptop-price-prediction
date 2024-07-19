import sys, os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Brand: str, Processor_Speed: float, RAM_Size: int, Storage_Capacity: int, Screen_Size: float, Weight: float):
        self.Brand = Brand
        self.Processor_Speed = Processor_Speed
        self.RAM_Size = RAM_Size
        self.Storage_Capacity = Storage_Capacity
        self.Screen_Size = Screen_Size
        self.Weight = Weight

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Brand": [self.Brand],
                "Processor_Speed": [self.Processor_Speed],
                "RAM_Size": [self.RAM_Size],
                "Storage_Capacity": [self.Storage_Capacity],
                "Screen_Size": [self.Screen_Size],
                "Weight": [self.Weight],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
