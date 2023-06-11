
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from pyngrok import ngrok

# Load the trained model and preprocessing objects
model = joblib.load("final_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    attendance = float(request.form['attendance'])
    mid1 = float(request.form['mid1'])
    mid2 = float(request.form['mid2'])
    quiz1 = float(request.form['quiz1'])
    quiz2 = float(request.form['quiz2'])
    quiz3 = float(request.form['quiz3'])
    final = float(request.form['final'])

    # Preprocess the inputs
    data = pd.DataFrame([[attendance, mid1, mid2, quiz1, quiz2, quiz3, final]], columns=['Attendance', 'Mid I', 'Mid II', 'Quiz 1', 'Quiz 2', 'Quiz 3', 'Final Exam'])
    data = scaler.transform(data)

    # Make the grade prediction
    prediction = model.predict(data)
    predicted_grade = encoder.inverse_transform(prediction)[0]
    print(predicted_grade)
    # Render the result template with the predicted grade
    return render_template('result.html', predicted_grade=predicted_grade)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

# Set up the ngrok tunnel
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)
