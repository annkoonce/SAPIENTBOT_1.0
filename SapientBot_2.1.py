# ============================
# IMPORT LIBRARIES
# ============================
import os
import discord
from discord.ext import commands
import openai
import json
import logging
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# ============================
# CONFIGURATION
# ============================
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
AI_KEY = os.getenv("AI_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
openai.api_key = AI_KEY

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

logging.basicConfig(level=logging.INFO)
debug_mode = False
q_table = {}  # Q-table for RL
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Define session_memory for storing user message history
session_memory = {}  # A dictionary to store user message history

# Predefined training data for fallback responses
training_data = {
    "greeting": [
        "Hello! How can I help?",
        "Hi there!",
        "Greetings! What can I do for you?",
        "Good morning! Ready to dive into something fun?",
        "Hey! What can I do for you?"
    ],
    "farewell": [
        "Goodbye! Take care!",
        "See you later! Have a great day!",
        "Farewell! Reach out anytime you need help.",
        "Bye for now! Stay awesome.",
        "Catch you later! Don’t hesitate to return."
    ],
    "video_games": [
        "Video games are a fantastic way to explore new worlds! What’s your favorite?",
        "Are you into RPGs, FPS games, or something else entirely?",
        "Minecraft is one of the most creative games ever made! What do you enjoy building?",
        "Looking for a new adventure? Try The Legend of Zelda: Breath of the Wild!",
        "Call of Duty is a classic! Do you prefer Warzone or multiplayer?"
    ],
    "general": [
        "Interesting, tell me more!",
        "Why do you ask?",
        "That’s fascinating!"
    ]
}

# ============================
# HELPER FUNCTIONS
# ============================
def update_q_table(state, action, reward):
    """Update the Q-table with new rewards."""
    if state not in q_table:
        q_table[state] = {}
    if action not in q_table[state]:
        q_table[state][action] = 0
    current_q = q_table[state][action]
    q_table[state][action] = current_q + alpha * (
        reward + gamma * max(q_table[state].values(), default=0) - current_q
    )

async def query_openai(prompt):
    """Query OpenAI API for responses."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message["content"].strip(), "ChatGPT"
    except Exception as e:
        logging.error(f"Error querying OpenAI: {e}")
        return "I'm sorry, I couldn't process your request.", "ChatGPT"

def query_google_search(question):
    """Query Google Search for responses."""
    try:
        from googlesearch import search
        search_results = list(search(question, num_results=3))
        return "\n".join(search_results), "Google Search"
    except Exception as e:
        logging.error(f"Error with Google Search: {e}")
        return "Unable to fetch results from Google.", "Google Search"

def detect_and_translate(message, target_language="en"):
    """Detect the language and translate to target language."""
    try:
        detected_language = detect(message)
        if detected_language != target_language:
            translated_message = GoogleTranslator(source=detected_language, target=target_language).translate(message)
            return detected_language, translated_message
        return detected_language, message
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return "unknown", message

# Save and load Q-table
QTABLE_FILE = "q_table.json"
def save_q_table():
    with open(QTABLE_FILE, "w") as f:
        json.dump(q_table, f)

def load_q_table():
    global q_table
    try:
        with open(QTABLE_FILE, "r") as f:
            q_table = json.load(f)
    except FileNotFoundError:
        q_table = {}

# ============================
# EVENT HANDLERS
# ============================
@bot.event
async def on_ready():
    load_q_table()
    print(f"Logged in as {bot.user.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = message.author.id
    user_lang = "en"  # Default language for responses

    # Track user history
    if user_id not in session_memory:
        session_memory[user_id] = []
    session_memory[user_id].append(message.content)
    if len(session_memory[user_id]) > 5:
        session_memory[user_id].pop(0)

    # Detect and translate the message
    detected_language, translated_message = detect_and_translate(message.content, target_language=user_lang)

    # Determine the conversational state
    if "hello" in message.content.lower():
        state = "greeting"
    elif "help" in message.content.lower():
        state = "help"
    elif "bye" in message.content.lower() or "goodbye" in message.content.lower():
        state = "farewell"
    elif "sapientbot" in message.content.lower():
        state = "name_recognition"
    else:
        state = "general"

    # Handle responses
    response = ""
    source = ""
    if state == "name_recognition":
        response = "Yes, I am SapientBot! I'm here to assist you with anything you need."
    source = "Custom Response"
    if state in q_table and q_table[state]:
        action = max(q_table[state], key=q_table[state].get)
        response = action
        source = "Reinforcement Learning"
    else:
        response, source = await query_openai(message.content)

    # Debug mode: Add source details
    if debug_mode:
        response += f"\n\n[Source: {source}]"

    # Handle translation output
    if detected_language != user_lang:
        # Translate the response back into English if it's not already in English
        translated_response = GoogleTranslator(source=detected_language, target="en").translate(response)
        response = f"Original: {message.content}\n[Translated to {user_lang.upper()}: {translated_message}]\n{response}\n(Translated to EN: {translated_response})"

    await message.channel.send(response)

    # Allow bot commands to be processed
    await bot.process_commands(message)

# ============================
# COMMANDS
# ============================
@bot.command(name="feedback")
async def feedback(ctx, state: str, action: str, reward: int):
    """Provide feedback to update the Q-table."""
    update_q_table(state, action, reward)
    save_q_table()
    await ctx.send(f"Thank you! Q-table updated with reward {reward} for state '{state}' and action '{action}'.")

@bot.command(name="qtable")
async def show_q_table(ctx):
    """Display a truncated or summarized version of the Q-table."""
    if not q_table:
        await ctx.send("The Q-table is currently empty.")
    else:
        # Summarize the Q-table to avoid exceeding Discord's character limit
        summary = []
        for state, actions in q_table.items():
            action_summary = ", ".join(f"{action}: {reward}" for action, reward in list(actions.items())[:3])
            summary.append(f"{state}: {action_summary} ...")
            # Add "..." to indicate there are more actions if truncated

        # Join all summaries and check length
        output = "\n".join(summary)
        if len(output) > 2000:
            output = output[:1997] + "..."  # Truncate to fit within the Discord limit

        await ctx.send(f"Q-Table (truncated):\n{output}")

@bot.command(name="debug")
async def toggle_debug(ctx, mode: str):
    """Toggle debug mode."""
    global debug_mode
    debug_mode = mode.lower() == "on"
    await ctx.send(f"Debug mode {'enabled' if debug_mode else 'disabled'}.")

# ============================
# RUN THE BOT
# ============================
if DISCORD_TOKEN:
    bot.run(DISCORD_TOKEN)
else:
    raise ValueError("DISCORD_TOKEN not set in environment variables.")
