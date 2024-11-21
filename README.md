# Sentiment Analysis Script and API

## Overview

This project includes two main components:

1.  **Sentiment Analysis Script (`main.py`)**: Benchmarks multiple algorithms for sentiment analysis on a dataset of customer reviews and saves the results to a CSV file.

2.  **Sentiment Analysis API (`api.py`)**: Provides a REST API for sentiment analysis and named entity recognition using pre-trained models.

3.  [OPTIONAL] A **R-Shiny app (`R_Shiny/app.R`)** to visualize the outputs of the models generated in **`main.py`** on any text provided by the user.

## Features

### Sentiment Analysis Script (`main.py`)

-   Uses pre-trained models for sentiment analysis:

    -   Hugging Face Transformers

    -   VADER (Valence Aware Dictionary and sEntiment Reasoner)

    -   TextBlob

-   Processes datasets with reviews.

-   Outputs a CSV file containing sentiment predictions from all three algorithms.

### Sentiment Analysis API (`api.py`)

-   Accepts text input via a RESTful interface.

-   Supports multiple tasks:

    -   Sentiment analysis (predicted by Hugging Face, VADER, and TextBlob).

    -   Placeholder for named entity recognition (NER).

-   Lightweight and deployable on Flask.

## Requirements

-   **Python 3.7+**

-   Install required libraries: pip install pandas flask transformers nltk textblob torch torchvision

-   For `transformers`, ensure PyTorch is properly installed based on your system configuration. Refer to PyTorch Installation ([https://pytorch.org/get-started/locally/).](https://pytorch.org/get-started/locally/).)

## Usage

### Running the Script (`main.py`)

1.  Prepare a CSV file containing a column named `review` with text data.

2.  Run the script: python main.py

3.  The output will be saved as `sentiment_results_benchmark.csv` in the current directory.

#### Output Example

The CSV file contains:

-   `review`: Original text.

-   `hf_sentiment`, `hf_score`: Sentiment and confidence from Hugging Face.

-   `vader_sentiment`, `vader_score`: Sentiment and compound score from VADER.

-   `textblob_sentiment`, `textblob_score`: Sentiment and polarity score from TextBlob.

### Running the API (`api.py`)

1.  Start the API: python api.py

2.  Access the API at [http://127.0.0.1:5000.](http://127.0.0.1:5000.)

#### API Endpoints

**Endpoint**: /predict\
**Method**: POST\
**Request Body** (JSON): { "text": "I love this movie!", "task": "sentiment" }

**Response Example**: { "huggingface": {"label": "POSITIVE", "score": 0.9998}, "vader": {"label": "positive", "score": 0.87}, "textblob": {"label": "positive", "score": 0.8} }

#### Testing

You can test the API using:

-   `curl`: curl -X POST "[http://127.0.0.1:5000/predict"](http://127.0.0.1:5000/predict%22) -H "Content-Type: application/json" -d '{"text": "I love this!", "task": "sentiment"}'

-   Postman or any REST client.

## Deployment

### Script (`main.py`)

-   Run directly in any Python environment.

### API (`api.py`)

-   **Flask with Gunicorn**: gunicorn -w 4 -b 0.0.0.0:8000 api:app

-   **Docker**: Create a Dockerfile to containerize the API for easy deployment.

## Customization

1.  **Add More Models**: Extend the script and API to include additional algorithms or models.

2.  **Improve API Functionality**: Add endpoints for tasks like entity recognition or topic classification.
