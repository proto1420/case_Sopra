library(shiny)
library(reticulate)
library(tidyverse)
library(plotly)

# Specify the virtual environment with PyTorch
use_virtualenv(here::here("torch-env/"), required = TRUE)

# Import required Python libraries
transformers <- import("transformers")
nltk <- import("nltk")
textblob <- import("textblob")

# Download necessary NLTK data
nltk$download("vader_lexicon")
vader <- nltk$sentiment$SentimentIntensityAnalyzer()

# Initialize Hugging Face pipeline (uses PyTorch backend by default)
pipeline <- transformers$pipeline
huggingface_model <- pipeline("sentiment-analysis", framework="pt")

# Helper functions
huggingface_sentiment <- function(text) {
  result <- huggingface_model(text)[[1]]
  list(label = result$label, score = result$score)
}

vader_sentiment <- function(text) {
  scores <- vader$polarity_scores(text)
  label <- if (scores$compound >= 0.05) "positive" else "negative"
  list(label = label, score = scores$compound)
}

textblob_sentiment <- function(text) {
  analysis <- textblob$TextBlob(text)
  label <- if (analysis$sentiment$polarity > 0) "positive" else "negative"
  list(label = label, score = analysis$sentiment$polarity)
}

# Define UI
ui <- fluidPage(
  titlePanel("Sentiment Analysis Visualizer"),
  sidebarLayout(
    sidebarPanel(
      textInput("input_text", "Enter Text:", value = "I love this movie!"),
      actionButton("analyze", "Analyze Sentiment")
    ),
    mainPanel(
      plotlyOutput("sentimentPlot"),
      verbatimTextOutput("resultText")
    )
  )
)

# Define Server Logic
server <- function(input, output) {
  # Reactive to store the results
  sentimentResults <- reactiveValues(
    huggingface = NULL,
    vader = NULL,
    textblob = NULL
  )

  # Analyze sentiment when the button is clicked
  observeEvent(input$analyze, {
    input_text <- input$input_text

    # Run sentiment analysis using all three methods
    sentimentResults$huggingface <- huggingface_sentiment(input_text)
    sentimentResults$vader <- vader_sentiment(input_text)
    sentimentResults$textblob <- textblob_sentiment(input_text)
  })

  # Render the sentiment plot
  output$sentimentPlot <- renderPlotly({
    req(sentimentResults$huggingface, sentimentResults$vader, sentimentResults$textblob)

    # Create a data frame for visualization
    results_df <- data.frame(
      Algorithm = c("Hugging Face", "VADER", "TextBlob"),
      Score = c(
        sentimentResults$huggingface$score,
        sentimentResults$vader$score,
        sentimentResults$textblob$score
      )
    )

    # Create the bar plot
    ggplotly(ggplot(results_df, aes(x = Algorithm, y = Score, fill = Algorithm)) +
      geom_bar(stat = "identity") +
      ylim(-1, 1) +
      labs(
        title = "Sentiment Scores from Different Algorithms",
        x = "Algorithm",
        y = "Sentiment Score"
      ) +
      theme_minimal())
  })

  # Render the result text
  output$resultText <- renderPrint({
    req(sentimentResults$huggingface,
        sentimentResults$vader,
        sentimentResults$textblob)

    list(
      HuggingFace = sentimentResults$huggingface,
      VADER = sentimentResults$vader,
      TextBlob = sentimentResults$textblob
    )
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
