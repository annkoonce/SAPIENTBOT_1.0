# intent_recognition.py

import spacy
import random

# Initialize spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Define intents and sample phrases
intent_phrases = {
    "greeting": ["hello", "hi", "hey", "greetings", "howdy"],
    "farewell": ["bye", "goodbye", "see you", "farewell"],
    "help": ["help", "assist", "support", "how to", "instructions", "guide"]
}

# Recognize intent
def recognize_intent(text):
    doc = nlp(text.lower())
    for intent, phrases in intent_phrases.items():
        if any(phrase in doc.text for phrase in phrases):
            return intent
    return "unknown"

# Generate response based on intent
def generate_response(intent):
    responses = {
        "greeting": ["Hello! How can I assist you today?", "Hi there! Need help with anything?"],
        "farewell": ["Goodbye! Take care!", "See you later!"],
        "help": ["I'm here to assist! Just ask me anything.", "You can ask about commands or for general assistance."],
        "unknown": ["I'm not sure I understand. Could you rephrase?"]
    }
    return random.choice(responses.get(intent, responses["unknown"]))
