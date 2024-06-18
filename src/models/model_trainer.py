import joblib
import logging
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.logger import CustomLogger, create_log_path


MODULE_NAME = 'model_trainer'
log_file_path = create_log_path(MODULE_NAME)
logger = CustomLogger(logger_name=MODULE_NAME,log_filename=log_file_path)
logger.set_log_level(level=logging.INFO)


def read_yaml(file_path):
    try:
        with open(file_path) as f:
            data = safe_load(f)
        return data
    except FileNotFoundError as e:
        logger.save_logs(msg=f"Error occured : {e}",log_level='error')

def save_model(model,file_name):
    joblib.dump(model,file_name)

def train_model():
    root_path = Path(__file__).parent.parent.parent
    models_path = root_path/'models'/'trained_models'
    models_path.mkdir(exist_ok=True)
    logger.save_logs(msg="trained_models folder created ")

    data_path = root_path/'data'/'processed'/'train.csv'
    data = pd.read_csv(data_path)
    logger.save_logs(msg="Data read successfully")

    yaml_file = read_yaml('params.yaml')
    logger.save_logs(msg="yaml file read successfully")

    params = yaml_file['model_params']
    models = {
        'linear_regression' : LinearRegression(),
        'ridge': Ridge(),
        'xgboost': XGBRegressor(**params['xgboost']),
        'random_forest' : RandomForestRegressor(**params['random_forest']),
        'lightgbm': LGBMRegressor(**params['lightgbm'])
    }

    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]


    for name,model in models.items():
        model.fit(X_train,y_train)
        save_model(model,models_path/(name + '.joblib'))
        logger.save_logs(msg=f"{name} models successfully trained and saved")


if __name__ == "__main__":
    train_model()