from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    if len(int_features) < 7:
        return render_template('home.html', pred='Please fill all the fields with correct Data',
                               bhai="kuch karna hain iska ab?")
    else:
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('home.html',
                               pred='Student will Pass the exams'.format(output)
                               , bhai="kuch nai karna hain iska ab?")
    else:
        return render_template('home.html',
                               pred='Student will fail the exams '
                               , bhai="kuch karna hain iska ab? Parh loo thora")


if __name__ == '__main__':
    app.run(debug=True)
