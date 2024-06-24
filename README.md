# Combat-Online-Plagiarism-with-AI
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Sample text data
texts = ["This is a sample text for plagiarism detection.", "Another sample text with different words."]

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    similarity = cosine_similarity([processed_text1], [processed_text2])
    return similarity[0][0]

@app.route('/plagiarism', methods=['POST'])
def plagiarism_check():
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({'error': 'Missing text input'}), 400

    max_similarity = 0
    plagiarized_text = None

    for text in texts:
        similarity = calculate_similarity(input_text, text)
        if similarity > max_similarity:
            max_similarity = similarity
            plagiarized_text = text

    return jsonify({'similarity': max_similarity, 'plagiarized_text': plagiarized_text})

if __name__ == '__main__':
    app.run(debug=True)
