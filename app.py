from flask import Flask, json, request, jsonify, render_template
import joblib
from flask_cors import CORS, cross_origin
# from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
# def basic():
#     return jsonify({'say':'hello'})
def basic():
    return render_template('index.html')

@app.route("/submit",methods=["GET","POST"])
def submit():
        try:
            algo = joblib.load('model/logistic_regression_model.pkl')
            vectorizer = joblib.load('model/vvv_model.pkl')
            text = request.form.get('words', default='')
            # Transform the input using CountVectorizer
            if vectorizer is not None:
                word_transformed = vectorizer.transform([text])

                # Make predictions using the loaded model
                y_pred = algo.predict(word_transformed)
                if y_pred == 0:
                    word = 'Tidak Ilmiah'
                    return render_template('result.html', prediction=word)
                else:
                    word = 'Ilmiah'
                    return render_template('result.html', prediction=word)
            else:
                return {"error"}
        
        except Exception as e:
            return {"error ":str(e)}

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(port=5000)