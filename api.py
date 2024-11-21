from flask import Flask, request, jsonify
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize models
huggingface_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize Flask app
app = Flask(__name__)

# Helper functions
def huggingface_sentiment(text):
    result = huggingface_model(text[:512])[0]
    return {"label": result['label'], "score": result['score']}

def vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    return {"label": "positive" if scores['compound'] >= 0.05 else "negative", "score": scores['compound']}

def textblob_sentiment(text):
    analysis = TextBlob(text)
    return {"label": "positive" if analysis.sentiment.polarity > 0 else "negative", "score": analysis.sentiment.polarity}
  

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()

    # Validate input
    if not data or "text" not in data or "task" not in data:
        return jsonify({"error": "Invalid input. Provide 'text' and 'task' fields."}), 400

    text = data["text"]
    task = data["task"].lower()

    if task == "sentiment":
        hf_result = huggingface_sentiment(text)
        vader_result = vader_sentiment(text)
        tb_result = textblob_sentiment(text)
        return jsonify({
            "huggingface": hf_result,
            "vader": vader_result,
            "textblob": tb_result
        })
    else:
        return jsonify({"error": "Invalid task. Only 'sentiment' is implemented."}), 400

if __name__ == '__main__':
    app.run(debug=True)
    
    
## Test the API
# curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"text": "I love Python!", "task": "sentiment"}'

