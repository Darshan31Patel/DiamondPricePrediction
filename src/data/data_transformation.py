import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.logger import CustomLogger,create_log_path
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODULE_NAME = "data_transformation"
log_file_path = create_log_path(MODULE_NAME)

logger = CustomLogger(logger_name=MODULE_NAME,log_filename=log_file_path)
logger.set_log_level(level=logging.INFO)

def create_preprocessor():
    try:
        categorical_col = ['cut', 'color','clarity']
        numerical_col =  ['carat', 'depth','table', 'x', 'y', 'z']
        target_col = ['price']

        cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
        color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

        cat_pipeline = Pipeline([
            ('ordinal-encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
            ('standard-scaler',StandardScaler())
        ])
        num_pipeline = Pipeline([
            ('standard-scaler',StandardScaler())
        ])
        output_pipeline = Pipeline([
            ('output',PowerTransformer(method='box-cox'))
        ])

        preprocessor = ColumnTransformer([
            ('cat_pipeline',cat_pipeline,categorical_col),
            ('num_pipeline',num_pipeline,numerical_col),
            ('output',output_pipeline,target_col)
        ])

        logger.save_logs(msg="Preprocessor created")
    except:
        logger.save_logs(msg="Error occured",log_level='error')
    return preprocessor


def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / 'data'/'interim'
    processed_path = root_path/'data'/'processed'
    processed_path.mkdir(exist_ok=True)

    logger.save_logs(msg=f'processed file created at {processed_path}')

    train_df = pd.read_csv(data_path/'train.csv')
    test_df = pd.read_csv(data_path/'test.csv')

    logger.save_logs(msg="Train and test file loaded succesfully")

    preprocessor = create_preprocessor()
    logger.save_logs(msg="preprocessor object loaded succesfully")

    preprocessor.set_output(transform='pandas')
    train_preprocess = preprocessor.fit_transform(train_df)
    test_preprocess = preprocessor.transform(test_df)

    preprocessor_path = root_path/'models'/'preprocessor.pkl'
    joblib.dump(preprocessor,preprocessor_path)

    logger.save_logs(msg="Data preprocessed succesfully")

    train_preprocess.to_csv(processed_path/'train.csv',index=False)
    logger.save_logs(msg="Train data saved succesfully")
    test_preprocess.to_csv(processed_path/'test.csv',index=False)
    logger.save_logs(msg="Test data saved succesfully")



if __name__=="__main__":
    main()