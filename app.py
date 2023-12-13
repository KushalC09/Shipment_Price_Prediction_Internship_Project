
import pickle

from flask import Flask,request,jsonify,render_template
import numpy as np

from sklearn.linear_model import LinearRegression


app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('K:\Major project\Shipment_Price_Prediction_Internship-master\lgbm_tuned3.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Shipment price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
