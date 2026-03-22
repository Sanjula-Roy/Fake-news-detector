from flask import Flask, render_template, request
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model + tfidf
with open("fake_news_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = joblib.load(f)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    title = request.form["title"]
    news = request.form["news"]

    cleaned = clean_text(title + " " + news)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]

    result = "REAL NEWS ✔️" if pred == 1 else "FAKE NEWS ❌"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)