# ============================
# IMPORT LIBRARIES
# ============================
import os
import discord
import spacy
import spotipy
import threading
import requests
import random
import atexit
import logging
import sqlite3
from datetime import datetime, timedelta
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from dotenv import load_dotenv
from spacy.matcher import Matcher
from textblob import TextBlob
from discord.ext import commands
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import asyncio

# Import custom modules
from response_handler import get_response
from state_manager import get_user_state, set_user_state, clear_user_state
from training_data import TRAINING_DATA
from intent_recognition import recognize_intent
from dialogue_state_machine import DialogueStateMachine
from gamification import GamificationSystem

# ============================
# Load environment variables and initialize bot
# ============================
load_dotenv()
OPENAO_TOKEN = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

# Set up Discord bot with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# In-memory storage for user preferences and feedback
user_preferences = {}
user_feedback = {}

# Initialize spaCy and matcher
nlp = spacy.blank("en")
matcher = Matcher(nlp.vocab)

# Initialize the chatbot
chatbot = ChatBot(
    'SAPIENTBOT',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3'
)

# Set up a trainer for the bot
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

# Initialize Dialogue State Machine
dialogue_state_machine = DialogueStateMachine()

# ============================
# Database Connection
# ============================
def connect_db():
    conn = sqlite3.connect('sapientbot.db')  # Path to your database file
    cursor = conn.cursor()
    return conn, cursor

# ============================
# User Preferences Functions
# ============================
def set_user_preference_in_memory(user_id, key, value):
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    user_preferences[user_id][key] = value

def get_user_preference_in_memory(user_id, key):
    if user_id in user_preferences and key in user_preferences[user_id]:
        return user_preferences[user_id][key]
    return None

# ============================
# Feedback Handling
# ============================
def save_feedback_in_memory(user_id, query, response, feedback):
    if user_id not in user_feedback:
        user_feedback[user_id] = []
    user_feedback[user_id].append({
        "query": query,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    })

# ============================
# Gamification System
# ============================
gamification = GamificationSystem()

# ============================
# Sentiment Analysis Function
# ============================
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "positive"
    elif analysis.sentiment.polarity < -0.1:
        return "negative"
    return "neutral"

# ============================
# Conversational AI Flow Function
# ============================
async def conversational_response(ctx, user_input):
    # Recognize the intent of the user input
    intent = recognize_intent(user_input)
    user_state = get_user_state(ctx.author.id)
    response = ""

    # Use the state machine to handle dialogue flow
    if user_state:
        response = dialogue_state_machine.continue_dialogue(user_state, user_input)
    else:
        if intent == "greeting":
            response = random.choice(["Hello! How can I assist you today?", "Hi there! Need any help?", "Greetings! What can I do for you?"])
        elif intent == "farewell":
            response = random.choice(["Goodbye! Feel free to reach out if you need anything else.", "See you later!", "Take care, and have a great day!"])
        elif intent == "help":
            response = "I'm here to assist you! You can ask me anything, like setting preferences, getting information, or just for a friendly chat."
        else:
            response = get_response(user_input)

    # Save the user state and respond
    set_user_state(ctx.author.id, dialogue_state_machine.current_state)
    await ctx.send(response)

# ============================
# Bot Commands
# ============================
@bot.command(name='hello')
async def hello(ctx):
    await conversational_response(ctx, "hello")

@bot.command(name='set_preference')
async def set_preference_command(ctx, key: str, value: str):
    set_user_preference_in_memory(ctx.author.id, key, value)
    await ctx.send(f"Preference '{key}' set to '{value}'.")

@bot.command(name='get_preference')
async def get_preference_command(ctx, key: str):
    value = get_user_preference_in_memory(ctx.author.id, key)
    if value:
        await ctx.send(f"Your preference for '{key}' is '{value}'.")
    else:
        await ctx.send(f"No preference set for '{key}'.")

@bot.command(name='analyze_feedback')
@commands.has_role('Admin')
async def analyze_feedback_command(ctx):
    user_id = str(ctx.author.id)
    if user_id not in user_feedback or not user_feedback[user_id]:
        await ctx.send("No feedback recorded yet.")
        return
    for feedback in user_feedback[user_id][-10:]:
        await ctx.send(
            f"Query: {feedback['query']}\n"
            f"Response: {feedback['response']}\n"
            f"Feedback: {feedback['feedback']}\n"
            f"Timestamp: {feedback['timestamp']}"
        )

@bot.command(name='profile')
async def profile(ctx):
    user_id = str(ctx.author.id)
    total_points = gamification.get_user_rewards(user_id)
    level = gamification.get_user_level(user_id)
    badges = gamification.get_badges(user_id)
    badge_str = ", ".join(badges) if badges else "No badges yet."
    await ctx.send(f"Here’s your profile, {ctx.author.name}:\n"
                   f"**Points**: {total_points}\n"
                   f"**Level**: {level}\n"
                   f"**Badges**: {badge_str}")

# ============================
# Fun Commands
# ============================
@bot.group(name='fun', invoke_without_command=True)
async def fun(ctx):
    await ctx.send(
        "Check out the fun commands!\n\n"
        "Commands:\n"
        "  8ball\n"
        "  flip\n"
        "  motivate\n"
        "  riddle\n"
        "  trivia\n"
        "  spotify\n\n"
        "Type `!help fun command` for more info on a command.\n"
        "You can also type `!help fun` for more info on this category."
    )

@fun.command(name='8ball')
async def magic_8ball(ctx, *, question: str):
    responses = [
        "It is certain.", "It is decidedly so.", "Without a doubt.",
        "Yes – definitely.", "You may rely on it.", "As I see it, yes.",
        "Most likely.", "Outlook good.", "Yes.", "Signs point to yes.",
        "Reply hazy, try again.", "Ask again later.", "Better not tell you now.",
        "Cannot predict now.", "Concentrate and ask again.",
        "Don't count on it.", "My reply is no.", "My sources say no.",
        "Outlook not so good.", "Very doubtful."
    ]
    await ctx.send(f"🎱 {random.choice(responses)}")

@fun.command(name='flip')
async def coin_flip(ctx):
    outcome = random.choice(['Heads', 'Tails'])
    await ctx.send(f"🪙 The coin landed on: **{outcome}**!")

@fun.command(name='motivate')
async def motivate(ctx):
    quotes = [
        "Believe you can and you're halfway there.",
        "It always seems impossible until it’s done.",
        "You are stronger than you think.",
        "Keep going, you're doing amazing.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts."
    ]
    await ctx.send(f"💪 {random.choice(quotes)}")

@fun.command(name='riddle')
async def riddle(ctx):
    riddles = {
        "I speak without a mouth and hear without ears. I have nobody, but I come alive with the wind. What am I?": "An echo",
        "The more of this there is, the less you see. What is it?": "Darkness",
        "I have keys but open no locks. I have space but no room. You can enter, but you can’t go outside. What am I?": "A keyboard"
    }
    riddle, answer = random.choice(list(riddles.items()))
    await ctx.send(f"🧩 {riddle}")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        response = await bot.wait_for('message', check=check, timeout=30.0)
        if response.content.lower() == answer.lower():
            await ctx.send("🎉 Correct! You're a riddle master!")
        else:
            await ctx.send(f"❌ Incorrect! The correct answer was: {answer}")
    except asyncio.TimeoutError:
        await ctx.send(f"⌛ Time's up! The correct answer was: {answer}")

# ============================
# Spotify Commands
# ============================
@fun.group(name='spotify', invoke_without_command=True)
async def spotify(ctx):
    await ctx.send(
        "🎶 Spotify commands available:\n"
        "- `!fun spotify play` to play a track\n"
        "- `!fun spotify pause` to pause the track\n"
        "- `!fun spotify current` to see the current track"
    )

@spotify.command(name='play')
async def play_spotify(ctx):
    try:
        token = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI).get_access_token()
        sp = spotipy.Spotify(auth=token['access_token'])
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']
            sp.start_playback(device_id=device_id)
            await ctx.send("▶️ Spotify playback started.")
        else:
            await ctx.send("No active Spotify devices found. Please open Spotify on a device and try again.")
    except Exception as e:
        await ctx.send(f"Error starting Spotify playback: {e}")

@spotify.command(name='pause')
async def pause_spotify(ctx):
    try:
        token = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI).get_access_token()
        sp = spotipy.Spotify(auth=token['access_token'])
        sp.pause_playback()
        await ctx.send("⏸️ Spotify playback paused.")
    except Exception as e:
        await ctx.send(f"Error pausing Spotify playback: {e}")

@spotify.command(name='current')
async def current_spotify(ctx):
    try:
        token = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI).get_access_token()
        sp = spotipy.Spotify(auth=token['access_token'])
        current_track = sp.current_playback()
        if current_track and current_track['is_playing']:
            track_name = current_track['item']['name']
            artist_name = current_track['item']['artists'][0]['name']
            await ctx.send(f"🎵 Now playing: '{track_name}' by {artist_name}")
        else:
            await ctx.send("No track is currently playing.")
    except Exception as e:
        await ctx.send(f"Error retrieving current playback: {e}")

# ============================
# Close Database Connection on Exit
# ============================
@atexit.register
def close_gamification_connection():
    print("Shutting down...")  # Example for closing resources if needed

# ============================
# Run the bot
# ============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(DISCORD_TOKEN)
