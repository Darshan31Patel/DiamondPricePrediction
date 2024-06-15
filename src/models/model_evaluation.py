import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.logger import CustomLogger,create_log_path


MODULE_NAME = "model_prediction"
log_file_path = create_log_path(MODULE_NAME)
logger = CustomLogger(MODULE_NAME,log_file_path)


def evaluate_model(data,model,preprocessor_path):
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    _,target_preprocessor = joblib.load(preprocessor_path)
    y_pred = model.predict(X_test)
    y_test_inv = target_preprocessor.inverse_transform(y_test)
    y_pred_inv = target_preprocessor.inverse_transform(y_pred)

    score = r2_score(y_test_inv,y_pred_inv)
    rmse_score = np.sqrt(mean_squared_error(y_test_inv,y_pred_inv))
    mae_score = mean_absolute_error(y_test_inv,y_pred_inv)

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metric("r2_score", score)
        mlflow.log_metric("rmse_score", rmse_score)
        mlflow.log_metric("mae_score", mae_score)
        mlflow.sklearn.log_model(model, "model")

def main():
    root_path = Path(__file__).parent.parent.parent
    preprocessor_path = root_path / 'models' / 'preprocessors.pkl'

    data_path = root_path/'data'/'processed'/'test.csv'
    data = pd.read_csv(data_path)

    model_list = os.listdir(root_path/'models'/'trained_models')
    for i in model_list:
        logger.save_logs(msg=f'{i} model loaded successfully')
        model = joblib.load(i)
        evaluate_model(data,model,preprocessor_path)
        logger.save_logs(msg=f'{i} model successfully evaluated and logged to mlflow')


if __name__=="__main__":
    main()