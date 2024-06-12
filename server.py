from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
loaded_model = joblib.load('tuned_gradient_boosting_model.pk1')

@app.route('/')
def home():
    return render_template('netflix_interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    imdb_vote = float(request.form['imdb_vote'])

    input_data = np.array([[imdb_vote]])
    predicted_score = loaded_model.predict(input_data)

    return render_template('netflix_interface.html', prediction=predicted_score[0])

if __name__ == '__main__':
    app.run(port=8070)