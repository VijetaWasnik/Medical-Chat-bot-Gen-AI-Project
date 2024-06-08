import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from flask import Flask, request, render_template

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Load pre-trained BERT model for QnA
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

# Load and preprocess PDF text
pdf_path = 'pdf/Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
preprocessed_text = preprocess_text(pdf_text)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer_question():
    question = request.form.get('question')
    
    if not question:
        return render_template('index.html', answer="No question provided", score=0)
    
    result = qa_pipeline({'question': question, 'context': preprocessed_text})
    return render_template('index.html', answer=result['answer'], score=result['score'])

if __name__ == '__main__':
    app.run(debug=True)
