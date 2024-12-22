import openai
import random
from textblob import TextBlob

# Session memory for context-aware conversations
session_memory = {}

# Function to initialize OpenAI API
def init_openai(api_key):
    openai.api_key = api_key

# Function to update session memory
def update_session_memory(user_id, message, max_messages=5):
    """
    Update session memory for a user. Keeps the last `max_messages` messages.
    """
    if user_id not in session_memory:
        session_memory[user_id] = []
    session_memory[user_id].append(message)
    if len(session_memory[user_id]) > max_messages:
        session_memory[user_id].pop(0)

# Function to generate an AI response using OpenAI
async def generate_ai_response(user_id, question):
    """
    Generate an AI-based response using session memory for context.
    """
    # Retrieve session memory for context
    context = "\n".join(session_memory.get(user_id, []))
    prompt = f"Conversation so far:\n{context}\nUser: {question}\nAI:"

    try:
        # Query OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the desired GPT model
            messages=[
                {"role": "system", "content": "You are a helpful conversational assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7
        )
        # Return OpenAI's response
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"


# Session memory for natural conversations
natural_conversation_memory = {}

async def generate_natural_response(user_id, message):
    """
    Generate a natural response based on user's message and conversation history.
    """
    # Get or initialize the user's conversation history
    conversation_history = natural_conversation_memory.get(user_id, [])
    conversation_history.append(f"User: {message}")

    # Create the conversation prompt
    prompt = "\n".join(conversation_history) + "\nSAPIENTBOT:"

    try:
        # Query OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # GPT-4 if available
            messages=[
                {"role": "system", "content": "You are a helpful and conversational assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract bot response
        bot_response = response['choices'][0]['message']['content'].strip()

        # Update the conversation history
        conversation_history.append(f"SAPIENTBOT: {bot_response}")
        natural_conversation_memory[user_id] = conversation_history[-10:]  # Keep only the last 10 exchanges

        return bot_response
    except Exception as e:
        return f"Error: {str(e)}"


# Sentiment analysis function
def analyze_sentiment(message):
    """
    Perform sentiment analysis on a message.
    """
    sentiment = TextBlob(message).sentiment
    return sentiment.polarity

# Function to handle user feedback
feedback_scores = {}

def collect_feedback(response_id, rating):
    """
    Collect and process feedback on a specific response.
    """
    if response_id not in feedback_scores:
        feedback_scores[response_id] = []
    feedback_scores[response_id].append(rating)
    avg_rating = sum(feedback_scores[response_id]) / len(feedback_scores[response_id])
    return avg_rating
