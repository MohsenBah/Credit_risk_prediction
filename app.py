import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os


app = Flask(__name__)


clf = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    a = [np.array(features)]
    b=pd.DataFrame(a,columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 
				'Checking account','Credit amount', 'Duration', 
				'Purpose'
				])
    c=pd.get_dummies(b,columns=[
                       'Sex', "Housing", 'Saving accounts', 'Checking account',
                       'Purpose'
                   ])
    new_x=pd.DataFrame(np.zeros((1,19)),columns=['Age', 'Job', 'Credit amount', 'Duration', 'Sex_male', 'Housing_own',
       'Housing_rent', 'Saving accounts_moderate',
       'Saving accounts_quite rich', 'Saving accounts_rich',
       'Checking account_moderate', 'Checking account_rich', 'Purpose_car',
       'Purpose_domestic appliances', 'Purpose_education',
       'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs',
       'Purpose_vacation/others'])
    for i in c.columns:
        if i in new_x.columns:
           new_x[i]=1
    pred = clf.predict(new_x)[0]
    if pred == 1:
       output = "non defaulter"
    else:
       output = "defaulter"


    return render_template('index.html', prediction_text='This customer is potentially a "{}".'.format(output))


if __name__ == "__main__":
    app.run(debug=True)