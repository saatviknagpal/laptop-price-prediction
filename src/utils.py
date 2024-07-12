import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import pickle
from exception import CustomException




def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as error:
        raise CustomException(error, sys)