# python -m venv tensorflow-env
# tensorflow-env\Scripts\activate
# pip install tensorflow transformers
# pip install typing-extensions==4.5.0

# python -m venv myenv
# myenv\Scripts\activate
# torch-env\Scripts\activate

import pandas as pd
from transformers import pipeline
from textblob import TextBlob
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load dataset
file_path = "C:/Users/APissoort/Documents/SopraSteria/Assessment/data/IMDB-movie-reviews.csv"

df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip', low_memory=False, delimiter=';')

assert 'review' in df.columns and 'sentiment' in df.columns, "Dataset must contain 'review' and 'sentiment' columns."

# Display the first few rows to verify the content
df.head()


# Initialize models
# huggingface_model = pipeline("sentiment-analysis")
huggingface_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
vader_analyzer = SentimentIntensityAnalyzer()

# Model functions
# def huggingface_sentiment(text):
#     """Classify sentiment using a Hugging Face model."""
#     result = huggingface_model(text[:512])[0]  # Truncate to model's max length
#     sentiment = "positive" if result['label'] == "POSITIVE" else "negative"
#     return sentiment, result['score']
# 
# def vader_sentiment(text):
#     """Classify sentiment using VADER."""
#     scores = vader_analyzer.polarity_scores(text)
#     sentiment = "positive" if scores['compound'] >= 0.05 else "negative"
#     return sentiment, scores['compound']
# 
# def textblob_sentiment(text):
#     """Classify sentiment using TextBlob."""
#     analysis = TextBlob(text)
#     sentiment = "positive" if analysis.sentiment.polarity > 0 else "negative"
#     return sentiment, analysis.sentiment.polarity
#   
  
def huggingface_sentiment(text):
    result = huggingface_model(text[:512])[0]
    return "positive" if result['label'] == "POSITIVE" else "negative"

def vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    return "positive" if scores['compound'] >= 0.05 else "negative"

def textblob_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative"

# Benchmarking function
def benchmark_model(model_name, sentiment_function, reviews, labels):
    start_time = time.time()
    predictions = [sentiment_function(review) for review in reviews]
    execution_time = time.time() - start_time
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", pos_label="positive")
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Execution Time (s)": execution_time
    }
    

# # Apply all models and benchmark
# results = []
# for review in df['review']:  # Ensure 'review' column exists in the dataset
#     try:
#         hf_sentiment, hf_score = huggingface_sentiment(review)
#         vader_sentiment_label, vader_score = vader_sentiment(review)
#         tb_sentiment, tb_score = textblob_sentiment(review)
#         results.append({
#             "review": review,
#             "hf_sentiment": hf_sentiment,
#             "hf_score": hf_score,
#             "vader_sentiment": vader_sentiment_label,
#             "vader_score": vader_score,
#             "textblob_sentiment": tb_sentiment,
#             "textblob_score": tb_score,
#         })
#     except Exception as e:
#         # Handle potential issues with specific rows
#         results.append({
#             "review": review,
#             "hf_sentiment": None,
#             "hf_score": None,
#             "vader_sentiment": None,
#             "vader_score": None,
#             "textblob_sentiment": None,
#             "textblob_score": None,
#             "error": str(e),
#         })
# 
# # Save results to a CSV file
# results_df = pd.DataFrame(results)
# output_path = "C:/Users/APissoort/Documents/SopraSteria/Assessment/output/sentiment_results.csv"
# results_df.to_csv(output_path, index=False)


# Benchmark all models
reviews = df['review'].tolist()
labels = df['sentiment'].tolist()  # Ensure labels are "positive" or "negative"

results = []
results.append(benchmark_model("Hugging Face", huggingface_sentiment, reviews, labels))
results.append(benchmark_model("VADER", vader_sentiment, reviews, labels))
results.append(benchmark_model("TextBlob", textblob_sentiment, reviews, labels))

# Convert results to a DataFrame for better visualization
benchmark_df = pd.DataFrame(results)

# Save results to a CSV file
output_path = "C:/Users/APissoort/Documents/SopraSteria/Assessment/output/sentiment_results.csv"
benchmark_df.to_csv(output_path, index=False)

# Display results
benchmark_df


print(f"Results saved to {output_path}")
