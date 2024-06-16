import os
import mlflow
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.logger import CustomLogger,create_log_path


MODULE_NAME = "model_evaluation"
log_file_path = create_log_path(MODULE_NAME)
logger = CustomLogger(MODULE_NAME,log_file_path)
logger.set_log_level(logging.INFO)

def evaluate_model(data,model,preprocessor_path):
    model_score = {}
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    _,target_preprocessor = joblib.load(preprocessor_path)
    y_pred = model.predict(X_test)

    y_test_df = pd.DataFrame(y_test, columns=['price'])
    y_test_inv = target_preprocessor.inverse_transform(y_test_df)

    y_pred_df = pd.DataFrame(y_pred, columns=['price'])
    y_pred_inv = target_preprocessor.inverse_transform(y_pred_df)

    score = r2_score(y_test_inv,y_pred_inv)
    rmse_score = np.sqrt(mean_squared_error(y_test_inv,y_pred_inv))
    mae_score = mean_absolute_error(y_test_inv,y_pred_inv)

    model_score['r2_score'] = score
    model_score['rmse'] = rmse_score
    model_score['mae'] = mae_score

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metric("r2_score", score)
        mlflow.log_metric("rmse_score", rmse_score)
        mlflow.log_metric("mae_score", mae_score)
        mlflow.sklearn.log_model(model, "model")

    return model_score

def main():
    root_path = Path(__file__).parent.parent.parent
    preprocessor_path = root_path / 'models' / 'preprocessors.pkl'

    data_path = root_path/'data'/'processed'/'test.csv'
    data = pd.read_csv(data_path)

    trained_model_path = root_path/'models'/'trained_models'
    model_list = os.listdir(trained_model_path)
    scores = {}
    # print(model_list)
    for i in model_list:
        logger.save_logs(msg=f'{i} model loaded successfully')
        model = joblib.load(trained_model_path/i)
        score = evaluate_model(data,model,preprocessor_path)
        logger.save_logs(msg=f'{i} model successfully evaluated and logged to mlflow')
        scores[i.split('.')[0]] = score

    sorted_models = sorted(scores.items(), key=lambda item: item[1]['r2_score'], reverse=True)
    # print(sorted_models)
    # print(sorted_models[0][0])

    best_model_name = sorted_models[0][0] + '.joblib'
    # print(best_model_name)
    best_model = joblib.load(trained_model_path/best_model_name)
    joblib.dump(best_model,root_path/'models'/'best_model')


if __name__=="__main__":
    main()