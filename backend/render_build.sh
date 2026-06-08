#!/bin/bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"