import random

# Define context dictionary to hold user-specific context
user_context = {}

# Define responses
RESPONSES = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! Need help with anything?", "Hey! What's up?"],
    "farewell": ["Goodbye! Take care!", "See you later!", "Farewell! Let me know if you need more help!"],
    "help": ["I'm here to assist! Just ask me anything.", "You can ask about commands or for general assistance.", "Need help? I'm right here for you!"],
    "thanks": ["You're welcome!", "Glad I could help!", "Happy to assist!"],
    "feedback_positive": ["Thank you for the feedback!", "I appreciate it!", "I'm glad you're happy with my help!"],
    "feedback_negative": ["Thank you for the feedback, I'll keep improving!", "I appreciate your input.", "I'll try to do better!"],
    "smalltalk_joke": ["Why did the chicken join a band? Because it had the drumsticks!", "What do you call fake spaghetti? An impasta!", "Why don’t scientists trust atoms? Because they make up everything!"],
    "smalltalk": ["I'm doing great, thanks for asking!", "All systems operational!", "I'm here to make your day easier!"],
    "capabilities": ["I can assist with commands, provide information, and have fun chats!", "I'm your helpful assistant here to make things easy!", "Ask me about commands or for help with tasks!"],
    "unknown": ["I'm not sure I understand. Could you rephrase?", "I'm here to help! Could you ask that differently?", "Let's try again – how can I help you?"]
}

def get_response(intent, user_id):
    # Check if there is context for the user
    previous_intent = user_context.get(user_id, "unknown")
    
    # Manage context flow
    if intent == "thanks" and previous_intent == "help":
        response = "Glad I could help! Is there anything else you need?"
    elif intent == "help" and previous_intent == "greeting":
        response = "Sure! Let me know what you need help with."
    else:
        # Default to random response for the intent
        response = random.choice(RESPONSES.get(intent, RESPONSES["unknown"]))
    
    # Update user context
    user_context[user_id] = intent
    
    return response
