import joblib
import pandas as pd
from pathlib import Path
from flask import Flask,render_template,request


current_path = Path(__file__).parent
model_path = current_path/'models'/'best_model.joblib'
preprocessor_path = current_path/'models'/'preprocessor.joblib'
input,target = joblib.load(preprocessor_path)
model = joblib.load(model_path)


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    result = None
       
    if request.method == "POST":
        cut = request.form.get('cut')
        color = request.form.get('color')
        clarity = request.form.get('clarity')
        carat =  request.form.get('carat')
        depth = request.form.get('depth')
        table =  request.form.get('table')
        x =  request.form.get('x')
        y =  request.form.get('y')
        z =  request.form.get('z')

        data_dict = {
            'cut':[cut],
            'color':[color],
            'clarity':[clarity],
            'carat': [float(carat) if carat else 0.0],
            'depth': [float(depth) if depth else 0.0],
            'table': [float(table) if table else 0.0],
            'x': [float(x) if x else 0.0],
            'y': [float(y) if y else 0.0],
            'z': [float(z) if z else 0.0],
            }
        data = pd.DataFrame(data_dict)
        scaled_data = input.transform(data)
        pred = model.predict(scaled_data)
        pred = pd.DataFrame(pred,columns=['price'])
        result = target.inverse_transform(pred)[0][0]
        result = round(result,0)

    return render_template("index.html",result=result)

if __name__=="__main__":
    app.run(host="localhost",port=3000)