from flask import Flask,render_template,request
import pickle 
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

app = Flask(__name__)

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i  in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()  
    for i in text:
     y.append(ps.stem(i))
    return " ".join(y)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    message = request.form['message']
    transformed = transform_text(message)
    vector_input = tfidf.transform([transformed])
    result  = model.predict(vector_input)[0]
    if result == 1:
     return render_template('index.html', prediction_text='Spam Message!')
    else:
     return render_template('index.html', prediction_text='Not Spam!')
if __name__ == '__main__':
 app.run(host='0.0.0.0',port=8080)