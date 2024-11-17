# ============================
# IMPORT LIBRARIES
# ============================
import os
import discord
import spacy
import spotipy
import threading
import webbrowser
import logging
import random
import unittest
import time
import json
import google
import requests
import praw
import openai
import atexit
import html
import sapientbot_db
import sqlite3

from multiprocessing import Process
from googlesearch import search
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from flask import Flask, request
from dotenv import load_dotenv
from spacy.matcher import Matcher
from textblob import TextBlob
from gamification import GamificationSystem
from discord.ext import commands
from google.cloud import dialogflow_v2 as dialogflow
from transformers import BertTokenizer, BertForSequenceClassification
from spacy.training import Example
from intent_recognition import recognize_intent, generate_response
from dialogue_state_machine import DialogueStateMachine
from gamification import GamificationSystem
from spacy.util import minibatch, compounding
from training_data import TRAINING_DATA
from response_handler import get_response
from spacy.training import Example
from state_manager import get_user_state, set_user_state, clear_user_state
from intent_recognizer import recognize_intent
from response_generator import generate_response, analyze_sentiment
from sapientbot_db import (add_or_update_user, get_user_data, update_points, get_user_badges, assign_badge, award_daily_bonus, get_leaderboard)
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from sapientbot_db import connect_db, update_points

def connect_db():
    conn = sqlite3.connect('sapientbot.db')  # Path to your database file
    cursor = conn.cursor()
    return conn, cursor
def set_user_preference_in_memory(user_id, key, value):
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    user_preferences[user_id][key] = value

def get_user_preference_in_memory(user_id, key):
    if user_id in user_preferences and key in user_preferences[user_id]:
        return user_preferences[user_id][key]
    return None

def save_feedback_in_memory(user_id, query, response, feedback):
    user_feedback.append({
        "user_id": user_id,
        "query": query,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    })

# ============================0
# Load environment variables and initialize bot
# ============================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')


sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Set up Discord bot with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# In-memory storage for user preferences and feedback
user_preferences = {}
user_feedback = []

# Initialize spaCy and matcher
nlp = spacy.load('en_core_web_sm')
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

# Train the chatbot on English corpus
trainer.train("chatterbot.corpus.english")

# Add a text classifier to the pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Add labels to the text classifier
intents = ["greeting", "farewell", "help"]
for intent in intents:
    textcat.add_label(intent)

# Train the model
def train_model(nlp, training_data, n_iter=20):
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        losses = {}
        
        # Create minibatches of the data
        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
                
            # Update the model with Example objects
            nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
        
        print(f"Iteration {i + 1}, Losses: {losses}")

    return nlp

# Train the model and save it
trained_nlp = train_model(nlp, TRAINING_DATA)
trained_nlp.to_disk("intent_model")

print("Model trained and saved to 'intent_model'")
# ============================
# Define Intents and Phrases
# ============================
intents_phrases = {
    "greeting": ["hello", "hi", "hey", "greetings", "howdy"],
    "farewell": ["bye", "goodbye", "see you", "farewell"],
    "help": ["help", "assist", "support", "how to", "instructions", "guide"],
    "smalltalk": ["how are you", "what’s up", "what’s going on", "how’s it going"]
}

# Add patterns to matcher for each intent
for intent, phrases in intents_phrases.items():
    patterns = [[{"LOWER": phrase.lower()}] for phrase in phrases]
    matcher.add(intent, patterns)

# Function to recognize intent
def recognize_intent(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    for match_id, start, end in matches:
        intent = nlp.vocab.strings[match_id]
        return intent
    return "unknown"

# Function to generate a response based on intent
def generate_response(intent):
    responses = {
        "greeting": ["Hello! How can I assist you today?", "Hi there! Need help with anything?", "Hey! What’s up?"],
        "farewell": ["Goodbye! Take care!", "See you later!", "Farewell! Let me know if you need more help!"],
        "help": ["I'm here to assist! Just ask me anything.", "You can ask about commands or for general assistance.", "Need help? I'm right here for you!"],
        "smalltalk": ["I'm just a bot, but I'm here to chat!", "All systems operational! How can I assist?", "I'm here to make your day easier!"],
        "unknown": ["I'm not sure I understand. Could you rephrase?", "I'm here to help! Could you ask that differently?", "Let's try again – how can I help you?"]
    }
    return random.choice(responses.get(intent, responses["unknown"]))
# ============================
# Handle all messages and commands
# ============================
user_context = {}
user_last_quiz_time = {}
QUIZ_COOLDOWN = timedelta(days=1)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if the user prefers formal responses
    preferred_tone = get_user_preference_in_memory(message.author.id, "tone")
    if preferred_tone == "formal":
        await message.channel.send("Good day to you. How may I assist?")
    else:
        await bot.process_commands(message)  # Ensure commands like !help work

# ============================
# Reinforcement Learning Module
# ============================
class RLModule:
    def __init__(self):
        self.q_table = {}  # Holds state-action values
        self.actions = ['reward_user', 'ignore', 'provide_tip']
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration factor
    
    def update_q_table(self, state, action, reward):
        # Initialize the Q-values for a new state
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}

        max_future_q = max(self.q_table[state].values())  # Get max Q-value for the next state
        current_q = self.q_table[state][action]
        
        # Q-learning formula
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
    
    def choose_action(self, state):
        # Exploration vs. exploitation (epsilon-greedy)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore: Random action
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit: Choose best action
        return random.choice(self.actions)  # Default to random action if state not in Q-table

# Create an RLModule instance
rl_module = RLModule()

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
# Commands
# ============================
@bot.command(name='hello')
async def hello(ctx):
    await ctx.send("Hello! How can I assist you today?")
    
async def set_preference_command(ctx, key: str, value: str):
    set_user_preference_in_memory(ctx.author.id, key, value)
    await ctx.send(f"Preference '{key}' set to '{value}'.")
    
async def get_preference_command(ctx, key: str):
    value = get_user_preference_in_memory(ctx.author.id, key)
    if value:
        await ctx.send(f"Your preference for '{key}' is '{value}'.")
    else:
        await ctx.send(f"No preference set for '{key}'.")
        
async def feedback_command(ctx, feedback: int, *, query: str):
    response = chatbot.get_response(query)
    save_feedback_in_memory(ctx.author.id, query, str(response), feedback)
    await ctx.send("Thank you for your feedback!")

@bot.command(name='analyze_feedback')
@commands.has_role('Admin')  # Ensure only admins can use this   
async def analyze_feedback_command(ctx):
    if not user_feedback:
        await ctx.send("No feedback recorded yet.")
        return

    for feedback in user_feedback[-10:]:  # Show the last 10 feedback entries
        await ctx.send(
            f"User: {feedback['user_id']}\n"
            f"Query: {feedback['query']}\n"
            f"Response: {feedback['response']}\n"
            f"Feedback: {feedback['feedback']}\n"
            f"Timestamp: {feedback['timestamp']}"
        )

# ============================
# Gamification System
# ============================
class GamificationSystem:
    def __init__(self):
        self.user_rewards = {}         # Stores user rewards (points)
        self.user_levels = {}          # Stores user levels
        self.badges = {}               # Stores badges earned by users
        self.last_reward_time = {}     # Track the last time each user was rewarded

    def reward_user(self, user_id, points):
        # Initialize user data if not present
        if user_id not in self.user_rewards:
            self.user_rewards[user_id] = 0
            self.user_levels[user_id] = 1
            self.badges[user_id] = []  # Initialize empty badge list
            self.last_reward_time[user_id] = datetime.now() - timedelta(days=1)  # Allow first bonus

        # Update user rewards and check for level-ups and badges
        self.user_rewards[user_id] += points
        self.check_level_up(user_id)
        self.assign_badges(user_id)

    def get_user_rewards(self, user_id):
        return self.user_rewards.get(user_id, 0)

    def get_user_level(self, user_id):
        return self.user_levels.get(user_id, 1)

    def check_level_up(self, user_id):
        points = self.user_rewards[user_id]
        new_level = points // 100  # Every 100 points = 1 level

        # Level up the user if their new level is higher
        if new_level > self.user_levels[user_id]:
            self.user_levels[user_id] = new_level
            return True
        return False

    def assign_badges(self, user_id):
        # Add a "500 Club" badge if points exceed 500 and badge isn't already assigned
        if user_id not in self.badges:
            self.badges[user_id] = []  # Ensure the badge list is initialized

        points = self.user_rewards[user_id]
        if points >= 500 and '500 Club' not in self.badges[user_id]:
            self.badges[user_id].append('500 Club')
    
    def get_badges(self, user_id):
        return self.badges.get(user_id, [])

    def daily_bonus(self, user_id):
        now = datetime.now()
        last_reward = self.last_reward_time.get(user_id, now - timedelta(days=1))

        if (now - last_reward).days >= 1:
            self.user_rewards[user_id] += 50  # Add daily bonus points
            self.last_reward_time[user_id] = now  # Update last reward time
            return True
        return False

gamification = GamificationSystem()

@bot.command(name='profile')
async def profile(ctx):
    user_id = str(ctx.author.id)
    user_data = gamification.get_user_data(user_id)
    if user_data:
        badge_list = ', '.join(user_data['badges']) if user_data['badges'] else 'No badges yet.'
        await ctx.send(
            f"{ctx.author.mention}, here is your profile:\n"
            f"Points: {user_data['points']}\n"
            f"Level: {user_data['level']}\n"
            f"Badges: {badge_list}"
        )
    else:
        await ctx.send(f"{ctx.author.mention}, you don’t have any points yet. Start interacting to earn some!")

# ============================
# Fun Commands
# ============================
# Define the Fun group
@bot.group(name='fun', invoke_without_command=True)
async def fun(ctx):
    """Check out fun commands to lighten up your day!"""
    await ctx.send(
        "Check out the fun commands!\n\n"
        "Commands:\n"
        "  8ball\n"
        "  flip\n"
        "  motivate\n"
        "  riddle\n"
        "  trivia\n\n"
        "Type `!help fun command` for more info on a command.\n"
        "You can also type `!help fun` for more info on this category."
    )
# 1. Magic 8-Ball
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

# 3. Dice Roll
@fun.command(name='roll')
async def roll_dice(ctx, dice: str):
    try:
        rolls, sides = map(int, dice.split('d'))
        results = [random.randint(1, sides) for _ in range(rolls)]
        await ctx.send(f"🎲 You rolled: {results} (Total: {sum(results)})")
    except ValueError:
        await ctx.send("Please use the format XdY (e.g., 2d6 for rolling two 6-sided dice).")

# 6. Trivia Game
# Trivia command
import html  # Import the html library to decode HTML entities

@bot.command(name='trivia')
async def trivia(ctx):
    url = "https://opentdb.com/api.php?amount=1&type=multiple"
    try:
        response = requests.get(url).json()
    except requests.RequestException as e:
        await ctx.send("Couldn't fetch a trivia question at the moment. Please try again later.")
        return

    if response['response_code'] == 0:
        question = html.unescape(response['results'][0]['question'])
        options = [html.unescape(opt) for opt in response['results'][0]['incorrect_answers']]
        correct_answer = html.unescape(response['results'][0]['correct_answer'])
        options.append(correct_answer)
        random.shuffle(options)

        options_str = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)])
        await ctx.send(f"🧠 Trivia Time!\n{question}\n\n{options_str}\n\nType the number of your answer.")

        def check(m):
            return m.author == ctx.author and m.content.isdigit()

        try:
            # Wait for the user's response without a time limit
            answer = await bot.wait_for('message', check=check)
            if options[int(answer.content) - 1] == correct_answer:
                points = 50
                user_id = str(ctx.author.id)
                update_points(user_id, points)
                user_data = get_user_data(user_id)
                total_points = user_data['points']
                await ctx.send(f"🎉 Correct! You've earned {points} points. Total Points: {total_points}")
            else:
                await ctx.send(f"❌ Incorrect! The correct answer was: {correct_answer}")
        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")
    else:
        await ctx.send("Couldn't fetch a trivia question. Try again later.")


# 8. Dad Joke Generator
@fun.command(name='dadjoke')
async def dad_joke(ctx):
    headers = {'Accept': 'application/json'}
    response = requests.get("https://icanhazdadjoke.com/", headers=headers).json()
    if 'joke' in response:
        await ctx.send(f"🤣 {response['joke']}")
    else:
        await ctx.send("Couldn't get a joke right now, try again later!")

# 9. Random Compliment
@fun.command(name='compliment')
async def compliment(ctx, *, member: discord.Member = None):
    if member is None:
        member = ctx.author
    compliments = [
        "You're an amazing person!", "You light up the room!", "Your smile is contagious.",
        "You have the best laugh!", "You bring out the best in other people."
    ]
    await ctx.send(f"{member.mention}, {random.choice(compliments)}! 😊")

# 10. Rock, Paper, Scissors Game
@fun.command(name='rps')
async def rock_paper_scissors(ctx, choice: str):
    bot_choice = random.choice(['rock', 'paper', 'scissors'])
    choice = choice.lower()
    if choice not in ['rock', 'paper', 'scissors']:
        await ctx.send("Please choose 'rock', 'paper', or 'scissors'.")
        return
    if choice == bot_choice:
        result = "It's a tie!"
    elif (choice == 'rock' and bot_choice == 'scissors') or \
        (choice == 'paper' and bot_choice == 'rock') or \
        (choice == 'scissors' and bot_choice == 'paper'):
        result = "You win! 🎉"
    else:
        result = "I win! 😎"
    await ctx.send(f"You chose {choice}, I chose {bot_choice}. {result}")

# 11. Quiz Game (Example of an additional command)
@fun.command(name='quiz')
async def quiz_command(ctx):
    # Quiz logic here (similar to previous code)
    await ctx.send("Quiz functionality is under the Fun group now!")
    
# Add Coin Flip to Fun group
@fun.command(name='flip')
async def coin_flip(ctx):
    outcome = random.choice(['Heads', 'Tails'])
    await ctx.send(f"🪙 The coin landed on: **{outcome}**!")

# Add Motivational Quotes to Fun group
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
# ============================
# Score Commands
# ============================ 
@bot.group(name='score', invoke_without_command=True)
async def score(ctx):
    """Check your scores and progress!"""
    await ctx.send_help(ctx.command)
    
@score.command(name='points')
async def points_command(ctx):
    user_id = ctx.author.id
    total_points = gamification.get_user_rewards(user_id)
    await ctx.send(f"You have {total_points} points!")

@score.command(name='profile')
async def profile_command(ctx):
    user_id = ctx.author.id
    total_points = gamification.get_user_rewards(user_id)
    level = gamification.get_user_level(user_id)
    badges = gamification.get_badges(user_id)
    badge_str = ", ".join(badges) if badges else "No badges yet."

    await ctx.send(f"Here’s your profile, {ctx.author.name}:\n"
                f"**Points**: {total_points}\n"
                f"**Level**: {level}\n"
                f"**Badges**: {badge_str}")

@score.command(name='leaderboard')
async def leaderboard_command(ctx):
    # Sort users by points in descending order and get the top 5
    top_users = sorted(gamification.user_rewards.items(), key=lambda x: x[1], reverse=True)[:5]
    
    leaderboard_text = "**Leaderboard**\n"
    for rank, (user_id, points) in enumerate(top_users, start=1):
        user = await bot.fetch_user(user_id)
        leaderboard_text += f"{rank}. {user.name} - {points} points\n"
    
    await ctx.send(leaderboard_text)
# ============================
# Voice Commands
# ============================
# Define the Voice group
@bot.group(name='voice', invoke_without_command=True)
async def voice(ctx):
    """🎙️ Voice commands to manage voice channel interactions."""
    await ctx.send("Available voice commands: join, leave. Use !voice <command> to manage the bot's presence in a voice channel.")

# Command to join the voice channel
@voice.command(name='join')
async def join_voice(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel}")
    else:
        await ctx.send("You need to be in a voice channel to use this command.")

# Command to leave the voice channel
@voice.command(name='leave')
async def leave_voice(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected from the voice channel.")
    else:
        await ctx.send("I'm not in a voice channel!")
# ============================
#Spotify Commands
# ============================
# Define the Music group for Spotify commands
@bot.group(name='music', invoke_without_command=True)
async def music(ctx):
    """🎶 Music commands for Spotify control."""
    await ctx.send("Available music commands: authorize, play, pause, current. Use !music <command> to control Spotify playback.")

# Spotify Commands
@bot.command(name='spotify_play')
async def spotify_play(ctx):
    try:
        token_info = sp_oauth.get_cached_token()
        if not token_info:
            await ctx.send("Please authenticate Spotify by visiting the URL provided.")
            auth_url = sp_oauth.get_authorize_url()
            webbrowser.open(auth_url)
            await ctx.send(f"Authenticate here: {auth_url}")
            return

        sp = spotipy.Spotify(auth=token_info['access_token'])
        sp.start_playback()
        await ctx.send(f"{ctx.author.mention}, started Spotify playback.")
    except Exception as e:
        await ctx.send(f"Error starting playback: {e}")

@bot.command(name='spotify_pause')
async def spotify_pause(ctx):
    try:
        token_info = sp_oauth.get_cached_token()
        if not token_info:
            await ctx.send("Please authenticate Spotify by visiting the URL provided.")
            auth_url = sp_oauth.get_authorize_url()
            webbrowser.open(auth_url)
            await ctx.send(f"Authenticate here: {auth_url}")
            return

        sp = spotipy.Spotify(auth=token_info['access_token'])
        sp.pause_playback()
        await ctx.send(f"{ctx.author.mention}, paused Spotify playback.")
    except Exception as e:
        await ctx.send(f"Error pausing playback: {e}")

@bot.command(name='spotify_now')
async def spotify_now_playing(ctx):
    try:
        token_info = sp_oauth.get_cached_token()
        if not token_info:
            await ctx.send("Please authenticate Spotify by visiting the URL provided.")
            auth_url = sp_oauth.get_authorize_url()
            webbrowser.open(auth_url)
            await ctx.send(f"Authenticate here: {auth_url}")
            return

        sp = spotipy.Spotify(auth=token_info['access_token'])
        current_track = sp.current_playback()
        if current_track and current_track['is_playing']:
            track_name = current_track['item']['name']
            artist_name = current_track['item']['artists'][0]['name']
            await ctx.send(f"Now playing: '{track_name}' by {artist_name}")
        else:
            await ctx.send("No track is currently playing.")
    except Exception as e:
        await ctx.send(f"Error retrieving current playback: {e}")
        
# ============================
# Logging & Debugging
# ============================     
logging.basicConfig(level=logging.INFO)

# Setup logging to write to a rotating file
log_handler = RotatingFileHandler('sapientbot_response_times.log', maxBytes=5*1024*1024, backupCount=5)
logging.basicConfig(handlers=[log_handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Track user command history
user_command_history = {}

def update_points(user_id, points):
    conn, cursor = connect_db()
    cursor.execute(
        "UPDATE users SET points = points + ? WHERE user_id = ?",
        (points, user_id)
    )
    conn.commit()
    cursor.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    new_points = cursor.fetchone()[0]
    print(f"Updated points for user {user_id}: {new_points}")  # Debugging line
    conn.close()

# ============================
# Close Database Connection on Exit
# ============================    
@atexit.register
def close_gamification_connection():
    gamification.close_connection()
# ============================
# Run the bot
# ============================
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)

