import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from data_models import PredictionDataset
from flask import Flask


# current_path = Path(__file__).parent
# model_path = current_path/'models'/'best_model.joblib'


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello"

if __name__=="__main__":
    app.run(host="localhost",port=3000)