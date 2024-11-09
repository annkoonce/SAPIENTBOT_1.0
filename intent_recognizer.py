
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define phrases for each intent
intents = {
    "joke": ["tell me a joke", "make me laugh", "something funny"],
    "trivia": ["play trivia", "trivia game", "let’s do trivia"],
    "gamification": ["check points", "my score", "leaderboard", "level"],
}

# Add patterns to matcher
for intent, phrases in intents.items():
    patterns = [[{"LOWER": word} for word in phrase.split()] for phrase in phrases]
    matcher.add(intent, patterns)

def recognize_intent(text):
    """Identify primary and secondary intents in user input."""
    doc = nlp(text.lower())
    matches = matcher(doc)
    
    detected_intents = []
    for match_id, start, end in matches:
        intent = nlp.vocab.strings[match_id]
        if intent not in detected_intents:
            detected_intents.append(intent)
            
    return detected_intents if detected_intents else ["unknown"]

