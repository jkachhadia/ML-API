from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from configs import *
from helper_functions import preprocess

app= Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return jsonify("hello from ML API of Titanic data!")

@app.route("/predictions", methods=["GET"])
def predictions():
    data = request.get_json()
    df=pd.DataFrame(data['data'])
    data_all_x_cols = cols
    try:
        preprocessed_df=preprocess(df)
    except:
        return jsonify("Error occured while preprocessing your data for our model!")
    filename=model_name
    loaded_model = pickle.load(open(filename, 'rb'))
    try:
        predictions= loaded_model.predict(preprocessed_df[data_all_x_cols])
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    response={'data':[],'prediction_label':{'survived':1,'not survived':0}}
    response['data']=list(predictions)
    return make_response(jsonify(response),200)

if __name__=='__main__':
    app.run(debug=True)