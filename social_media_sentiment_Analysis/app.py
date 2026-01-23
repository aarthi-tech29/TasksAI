# STEP 1: Import Required Libraries
import pandas as pd
from textblob import TextBlob

# pandas → handle text data
# TextBlob → sentiment analysis engine

# STEP 2: Create Sample Social Media Dataset
data = {
    "User": ["A", "B", "C", "D", "E"],
    "Post": [
        "I love this product so much",
        "This is the worst service ever",
        "It is okay, not bad",
        "Amazing experience and great support",
        "I am very disappointed"
    ]
}

df = pd.DataFrame(data)
print(df)

# Each row is one social media post.

# STEP 3: Sentiment Analysis Function
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# STEP 4: Apply Sentiment Function to Dataset

df["Sentiment"] = df["Post"].apply(get_sentiment)
print(df)

# STEP 5: Add Sentiment Score (Text Scoring)

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

df["Score"] = df["Post"].apply(get_sentiment_score)
print(df)

# STEP 6: Analyze Overall Sentiment

sentiment_count = df["Sentiment"].value_counts()
print(sentiment_count)

# STEP 7: User Input Sentiment Check (Like Console App)
while True:
    text = input("\nEnter a social media post (or 'exit'): ")

    if text.lower() == "exit":
        print("Exiting program.")
        break

    sentiment = get_sentiment(text)
    score = get_sentiment_score(text)

    print(f"Sentiment: {sentiment}")
    print(f"Score: {score}")
