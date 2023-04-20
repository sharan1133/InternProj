from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model


app = Flask(__name__)


#model = joblib.load('model.joblib')
ct = joblib.load('/home/sharanbalakrishnan/Desktop/InternProj/data/column_transformer.pkl')

#with open(r'/home/sharanbalakrishnan/Desktop/InternProj/data/my_rf.bin', 'rb') as f:
    #model = pickle.load(f)

model = tf.keras.models.load_model('/home/sharanbalakrishnan/Desktop/InternProj/data/my_model_tf.h5')



#color_mapping = {'blue': 0, 'black': 1, 'gold': 2, 'grey': 3, 'green': 4, 'white': 5, 'silver': 6,
                 #'yellow': 7, 'carbon': 8, 'purple': 9, 'orange': 10, 'pearl': 11, 'cream': 12}

#display_type_mapping = {'HD+': 0, 'AMOLED': 1, 'HD': 2, 'XDR': 3, 'Retina': 4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    ROM = int(request.form['ROM'])
    RAM = int(request.form['RAM'])
    Color = request.form['Color']
    Front_Cam = float(request.form['Front_Cam'])
    Rear_Cam = float(request.form['Rear_Cam'])
    Display_Size = float(request.form['Display_Size'])
    Display_Type = request.form['Display_Type']

    new_data = {
    'RAM': RAM,
    'ROM': ROM,
    'Color': Color,
    'Display_Type': Display_Type,
    'Display_Size': Display_Size,
    'Front_Cam': Front_Cam,
    'Rear_Cam': Rear_Cam
}

    new_df = pd.DataFrame([new_data])

    new_data = ct.transform(new_df)

    
    #prediction = model.predict([[storage, ram, color, front_cam, rear_cam, display, display_type]])
    prediction = model.predict(new_data)
    #predicted_mobile = prediction[0]
    predicted_mobile = float(prediction[0])
    
    #return render_template('result.html', result=predicted_mobile)
    #return render_template('index.html', prediction_text='Predicted Mobile Price: ₹ {:.2f}'.format(predicted_mobile))
    #return render_template('index.html', prediction_text=predicted_mobile)
    return render_template('index.html', prediction_text='Price is: ₹ {:.2f}'.format(predicted_mobile))


if __name__ == '__main__':
    app.run(debug=True)
