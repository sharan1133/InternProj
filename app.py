from flask import Flask, render_template, request
import pickle
import joblib

app = Flask(__name__)


#model = joblib.load('model.joblib')

with open(r'C:\Users\shara\OneDrive\Desktop\InternProj\my_rf.bin', 'rb') as f:
    model = pickle.load(f)


color_mapping = {'blue': 0, 'black': 1, 'gold': 2, 'grey': 3, 'green': 4, 'white': 5, 'silver': 6,
                 'yellow': 7, 'carbon': 8, 'purple': 9, 'orange': 10, 'pearl': 11, 'cream': 12}

display_type_mapping = {'HD+': 0, 'AMOLED': 1, 'HD': 2, 'XDR': 3, 'Retina': 4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    storage = int(request.form['storage'])
    ram = int(request.form['ram'])
    color = color_mapping[request.form['color']]
    front_cam = float(request.form['front_cam'])
    rear_cam = float(request.form['rear_cam'])
    display = float(request.form['display'])
    display_type = display_type_mapping[request.form['display_type']]
    
    
    prediction = model.predict([[storage, ram, color, front_cam, rear_cam, display, display_type]])
    predicted_mobile = prediction[0]
    
    #return render_template('result.html', result=predicted_mobile)
    return render_template('index.html', prediction_text='Predicted Mobile Price: ₹ {:.2f}'.format(predicted_mobile))

if __name__ == '__main__':
    app.run(debug=True)
