# response_generator.py
import random
from textblob import TextBlob

# Responses for each intent
responses = {
    "joke": [
        "Here's a joke for you! Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Why don’t scientists trust atoms? Because they make up everything!"
    ],
    "trivia": [
        "Ready for trivia? Type `!fun trivia` to start.",
        "Let’s test your knowledge! Try typing `!fun trivia`."
    ],
    "gamification": [
        "You can check your points with `!score points`.",
        "Want to see the leaderboard? Type `!score leaderboard`."
    ],
    "unknown": [
        "I'm not sure I understand. Could you rephrase?",
        "Let's try again – how can I help you?"
    ]
}

def analyze_sentiment(text):
    """Detect if user sentiment is positive, neutral, or negative."""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.1:
        return "positive"
    elif sentiment < -0.1:
        return "negative"
    return "neutral"

def generate_response(intent, user_sentiment="neutral"):
    """Generate a response based on intent and sentiment."""
    response_list = responses.get(intent, responses["unknown"])
    
    # Adjust tone based on sentiment
    if user_sentiment == "positive" and intent == "joke":
        return random.choice(response_list) + " 😂"
    elif user_sentiment == "negative" and intent == "unknown":
        return "I'm here to help – maybe try asking in a different way?"
    return random.choice(response_list)
