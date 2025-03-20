from tokenize import String
from flask import Flask, render_template,request,jsonify
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.json['email']
    print(list(str(data)))
    input_data=vectorizer.transform(list(str(data)))

    output=model.predict(input_data)[0]
    print(output)

    if output == 1:
        return jsonify("Spam")
    else:
        return jsonify("Not Spam")







if __name__=='__main__':
    app.run(debug=True)