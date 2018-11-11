
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getprediction',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form

        #Get form contents
        tv = result['trading_volume']
        change = result['change_in_percent']
        oc = result['Open-Close']
        hl = result['High-Low']

        #Prepare the feature vector for prediction
        pkl_file = open('var_predict', 'rb')
        index_dict = pickle.load(pkl_file)
        cat_vector = np.zeros(len(index_dict)).reshape(1,-1)
        
        try:
            cat_vector[index_dict['trading_volume'+str(tv)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['change_in_%'+str(change)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Open-Close'+str(oc)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['High-Low'+str(hl)]] = 1
        except:
            pass

        #Load model (SVC)
        pkl_file = open('model.pkl', 'rb')
        model = pickle.load(pkl_file)
        prediction = model.predict(cat_vector)
        
        return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
	app.debug = True
	app.run()