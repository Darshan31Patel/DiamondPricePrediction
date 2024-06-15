import logging
import joblib
import pandas as pd
from pathlib import Path
from src.logger import CustomLogger, create_log_path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODULE_NAME = "data_transformation"
log_file_path = create_log_path(MODULE_NAME)

logger = CustomLogger(logger_name=MODULE_NAME, log_filename=log_file_path)
logger.set_log_level(level=logging.INFO)

def create_preprocessors():
    try:
        categorical_col = ['cut', 'color', 'clarity']
        numerical_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
        # target_col = ['price']

        cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

        cat_pipeline = Pipeline([
            ('ordinal-encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
            ('standard-scaler', StandardScaler())
        ])
        num_pipeline = Pipeline([
            ('standard-scaler', StandardScaler())
        ])
        feature_preprocessor = ColumnTransformer([
            ('cat_pipeline', cat_pipeline, categorical_col),
            ('num_pipeline', num_pipeline, numerical_col)
        ])

        target_preprocessor = Pipeline([
            ('output', PowerTransformer(method='box-cox'))
        ])

        logger.save_logs(msg="Preprocessors created")
    except Exception as e:
        logger.save_logs(msg=f"Error occurred: {str(e)}", log_level='error')
        raise

    return feature_preprocessor, target_preprocessor

def main():
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / 'data' / 'interim'
    processed_path = root_path / 'data' / 'processed'
    processed_path.mkdir(exist_ok=True)

    logger.save_logs(msg=f'processed file created at {processed_path}')

    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    logger.save_logs(msg="Train and test files loaded successfully")

    feature_preprocessor, target_preprocessor = create_preprocessors()
    logger.save_logs(msg="Preprocessor objects loaded successfully")

    feature_preprocessor.set_output(transform='pandas')
    X_train_preprocessed = feature_preprocessor.fit_transform(train_df)
    y_train_preprocessed = target_preprocessor.fit_transform(train_df[['price']])
    X_test_preprocessed = feature_preprocessor.transform(test_df)
    y_test_preprocessed = target_preprocessor.transform(test_df[['price']])

    preprocessor_path = root_path / 'models' / 'preprocessors.pkl'
    joblib.dump((feature_preprocessor, target_preprocessor), preprocessor_path)

    logger.save_logs(msg="Data preprocessed successfully")

    train_preprocessed = pd.concat([X_train_preprocessed, pd.DataFrame(y_train_preprocessed, columns=['price'])], axis=1)
    test_preprocessed = pd.concat([X_test_preprocessed, pd.DataFrame(y_test_preprocessed, columns=['price'])], axis=1)

    train_preprocessed.to_csv(processed_path / 'train.csv', index=False)
    logger.save_logs(msg="Train data saved successfully")
    test_preprocessed.to_csv(processed_path / 'test.csv', index=False)
    logger.save_logs(msg="Test data saved successfully")

if __name__ == "__main__":
    main()
