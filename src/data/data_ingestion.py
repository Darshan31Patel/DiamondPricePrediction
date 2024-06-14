import logging
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import CustomLogger, create_log_path

MODULE_NAME = "data_ingestion"
log_file_path = create_log_path(MODULE_NAME)

logger = CustomLogger(logger_name=MODULE_NAME, log_filename=log_file_path)
logger.set_log_level(level=logging.INFO)

def extract_zip_file(input_path, output_path):
    try:
        with ZipFile(file=input_path) as f:
            f.extractall(path=output_path)
            input_file_name = input_path.name
            logger.save_logs(msg=f'{input_file_name} extracted successfully', log_level='info')
    except Exception as e:
        logger.save_logs(msg=f'Error extracting {input_path}: {e}', log_level='error')


def make_dataset(data_path):
    try:
        data = pd.read_csv(data_path)
        logger.save_logs(msg=f'Data successfully read from {data_path} shape : {data.shape}')
        train_data,test_data = train_test_split(data, test_size=0.25, random_state=42)
    except FileNotFoundError as e:
        logger.save_logs(msg="File not found",log_level='error')
    return train_data,test_data


def save_data(data:pd.DataFrame, path:Path):
    data.to_csv(path,index=False)
    logger.save_logs(msg=f'Data successfully saved at {path} shape : {data.shape}')


def main():
    current_path = Path(__file__).resolve()
    root_path = current_path.parent.parent.parent
    raw_data_path = root_path / 'data' / 'raw'
    input_data_path = raw_data_path / 'zipped' / 'data.zip'
    output_data_path = raw_data_path / 'extracted'
    output_data_path.mkdir(exist_ok=True, parents=True)

    extract_zip_file(input_path=input_data_path, output_path=output_data_path)

    extracted_data_path = output_data_path/'diamonds.csv'
    interim_data_path = root_path / 'data' / 'interim'
    interim_data_path.mkdir(exist_ok=True)
    
    train_data, test_data = make_dataset(extracted_data_path)
    save_data(data=train_data,path=interim_data_path/'train.csv')
    save_data(data=test_data, path=interim_data_path/'test.csv')

if __name__ == "__main__":
    main()