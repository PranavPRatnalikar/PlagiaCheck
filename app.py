from flask import Flask, render_template, request, send_file
from docx import Document
import re
from collections import Counter
from math import sqrt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import io
import base64

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def read_file_content(uploaded_file):
    if uploaded_file.filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.filename.endswith(".docx"):
        doc = Document(uploaded_file)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")


def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]


def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def calculate_word_similarity(text1, text2):
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    vec1 = Counter(words1)
    vec2 = Counter(words2)
    return cosine_similarity(vec1, vec2) * 100


def calculate_sentence_similarity(text1, text2):
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    similarities = []
    for sent1 in sentences1:
        max_similarity = 0
        for sent2 in sentences2:
            similarity = calculate_word_similarity(sent1, sent2)
            if similarity > max_similarity:
                max_similarity = similarity
        similarities.append(max_similarity)
    return sum(similarities) / len(similarities) if similarities else 0.0


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file1 = request.files["file1"]
            file2 = request.files["file2"]
            text1 = read_file_content(file1)
            text2 = read_file_content(file2)

            word_similarity = calculate_word_similarity(text1, text2)
            sentence_similarity = calculate_sentence_similarity(text1, text2)
            plagiarism_percentage = (word_similarity + sentence_similarity) / 2

            return render_template(
                "index.html",
                word_similarity=word_similarity,
                sentence_similarity=sentence_similarity,
                plagiarism_percentage=plagiarism_percentage,
                result=True,
            )
        except Exception as e:
            return render_template("index.html", error=str(e), result=False)

    return render_template("index.html", result=False)


if __name__ == "__main__":
    app.run(debug=True)
