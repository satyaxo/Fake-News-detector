from flask import Flask, render_template, request

app = Flask(__name__)

import string
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
df = pd.read_csv("df.csv")

# Load the pre-trained model
model = load_model("satyaaa.h5")  # Replace with the correct path

# Text Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['title'])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences)

# Streamlit app
def mai(news_text):
    # Assuming you already have 'model' trained

    # New input text for prediction
    new_texts = [news_text]

    # Preprocess the new texts
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])

    # Make predictions
    predictions = model.predict(new_padded_sequences)

    # Convert predictions to binary (0 or 1)
    binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    # Display the results
    for text, prediction in zip(new_texts, binary_predictions):
        print(f'Text: {text}\nPrediction: {"True" if prediction == 1 else "Fake"}\n')


    # Display result
    if prediction == 1:
        return 1
    else:
        return 0


@app.route('/', methods=['GET', 'POST'])
def index():
    xc = ''
    result = ''
    if request.method == 'POST':
        news_input = request.form['news_input']
        result = mai(news_input)
        xc = ''
        if result == 1:
            xc = 'true news'
        else:
            xc = 'false news'
    return render_template('index.html', result=xc)

if __name__ == '__main__':
    app.run(debug=True)