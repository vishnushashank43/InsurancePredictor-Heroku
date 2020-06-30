import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')





@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
       int_features = [float(x) for x in request.form.values()]
       final_features = [np.array(int_features)]
       prediction = model.predict(final_features)

    output = round(int(prediction[0]))
    age= int(int_features[0])
    sex=int(int_features[1])
    bmi=int(int_features[2])
    ch=int(int_features[3])
    smk=int(int_features[4])
    reg=int(int_features[5])
    return render_template('result.html',prediction_text=output,Age=age,sex =sex,bmi=bmi, smk=smk, ch=ch, reg=reg)
       
       

if __name__ == "__main__":
    app.run(debug=True)
    