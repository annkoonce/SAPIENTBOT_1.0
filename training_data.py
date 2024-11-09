
TRAINING_DATA = [
    ("Hello!", {"cats": {"greeting": 1.0, "farewell": 0.0, "help": 0.0}}),
    ("Hi there!", {"cats": {"greeting": 1.0, "farewell": 0.0, "help": 0.0}}),
    ("Goodbye!", {"cats": {"greeting": 0.0, "farewell": 1.0, "help": 0.0}}),
    ("See you later!", {"cats": {"greeting": 0.0, "farewell": 1.0, "help": 0.0}}),
    ("Can you help me?", {"cats": {"greeting": 0.0, "farewell": 0.0, "help": 1.0}}),
    ("I need assistance.", {"cats": {"greeting": 0.0, "farewell": 0.0, "help": 1.0}}),
    ("Thank you!", {"cats": {"thanks": 1.0}}),
    ("You're a great bot!", {"cats": {"feedback_positive": 1.0}}),
    ("This could be better.", {"cats": {"feedback_negative": 1.0}}),
    ("Tell me a joke!", {"cats": {"smalltalk_joke": 1.0}}),
    ("How are you?", {"cats": {"smalltalk": 1.0}}),
    ("What can you do?", {"cats": {"capabilities": 1.0}}),
    # Add more examples for other intents
]
