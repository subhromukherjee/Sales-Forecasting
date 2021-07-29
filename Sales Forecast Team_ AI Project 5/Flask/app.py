import numpy as np
from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
import pickle
import joblib

app = Flask(__name__)
model = load_model('Sales_Prediction_Forecast.h5')
scaler = joblib.load("scaler")

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/intro')
def intro():
    return render_template("intro.html")
@app.route('/predict')
def predict():
    return render_template("web.html")





@app.route('/login',methods = ['GET','POST'])
def login() :
    x_input=str(request.form['year'])
    x_input=x_input.split(',')
    for i in range(0, len(x_input)):
        x_input[i] = float(x_input[i])
    x_input=np.array(x_input).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=10
    i=0
    while(i<10):

        if(len(temp_input)>=10):
            x_input=np.array(temp_input[0:])
            print("{} day input {}".format(i,x_input))
            #x_input=np.expand_dims(x_input, axis=0)
            x_input=x_input.reshape(-1,1)
            
            x_input=scaler.transform(x_input)
            #x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print("x_input.....",x_input)
            yhat = model.predict(x_input, verbose=0)
            
            yhat=scaler.inverse_transform(yhat)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            print("Please give 10 number of inputs")

    return render_template("web.html",showcase = 'The next day predicted value is: '+str(round(lst_output[0][0],2)))

if __name__ == '__main__' :
    app.run(debug = True,port=5000)
