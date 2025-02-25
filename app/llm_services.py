import requests


def get_sentiment_explanation(tweet_text, sentiment):
    """
    Get an explanation for why the given sentiment was chosen for the tweet.

    Args:
        tweet_text (str): The original tweet text
        sentiment (str): The sentiment that was chosen

    Returns:
        str: The explanation for the sentiment choice
    """
    prompt = (
        f'Tweet: "{tweet_text}"\n'
        f"Sentiment: {sentiment}\n\n"
        "Task: Explain in 2-3 short sentences why this tweet was classified with the above sentiment.\n"
        "Be specific about the words, tone, or context that led to this classification.\n"
        "Response:"
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:latest", "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No explanation available.").strip()
    except Exception as e:
        print(f"Error getting explanation: {str(e)}")
        return "Unable to generate explanation."


def analyze_sentiment(tweet_text):
    """
    Analyze the sentiment of a tweet using zero-shot classification via Ollama.

    Args:
        tweet_text (str): The tweet text to analyze

    Returns:
        tuple: (sentiment, explanation) where sentiment is the predicted category
               and explanation is None (to be fetched separately)
    """
    # Construct a custom prompt using a zero-shot approach
    prompt = (
        "Task: Classify the sentiment of the following tweet into exactly one category.\n"
        "Categories: Positive, Negative, Neutral, Sarcastic, Funny, or Meme\n"
        "Rules:\n"
        "- Return ONLY the category name, nothing else\n"
        "- Choose the most appropriate category based on the tweet's content and tone\n"
        f'Tweet: "{tweet_text}"\n'
        "Category:"
    )

    try:
        # Make request to Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:latest", "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()

        # Extract and clean the sentiment from the response
        sentiment = result.get("response", "").strip()

        # Validate the sentiment is one of our categories
        valid_sentiments = {
            "Positive",
            "Negative",
            "Neutral",
            "Sarcastic",
            "Funny",
            "Meme",
        }
        if sentiment not in valid_sentiments:
            return "Neutral"  # Default to neutral if we get an invalid response

        return sentiment

    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return "Error analyzing sentiment"
