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
import openai
import random
import unittest
import time
import json
import google
import requests
import praw
# ============================
# FROM IMPORT LIBRARIES
# ============================
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
from discord.ext import commands


# ============================
# Load environment variables and initialize bot
# ============================
load_dotenv()
# ============================
# Bot Token and API keys
# ============================
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')
OPEN_API_ID = os.getenv('OPEN_API_ID')
# ============================
# Create bot object
# ============================
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
description = "I'm SapientBot! Here to help you with you all your questions, pay compliments and roll D20's. Please see the information below."
help_command = "!help"

user_greeted = set()

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
# Event listener for when a message is sent and the bot is mentioned
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore bot's own messages

    # Check if the bot is mentioned and if the user has a custom greeting
    if bot.user in message.mentions:
        if message.author.name in custom_greetings:
            # Check if the user has already been greeted to avoid multiple greetings
            if message.author.id not in user_greeted:
                await message.channel.send(custom_greetings[message.author.name])
                user_greeted.add(message.author.id)  # Add user to greeted set

    # Process commands if any are used
    await bot.process_commands(message)
# ============================
# Load NLP model
# ============================
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore the bot's own messages

    # Check if the message does not start with a command prefix
    if not message.content.startswith('!'):
        response = await generate_ai_response(message.content)
        await message.channel.send(response)
    else:
        await bot.process_commands(message)

# Handle Natural Language Understanding with spaCy
def handle_greetings(text):
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if nlp.vocab.strings[match_id] == "GREETING":
            return "Hello! How are you today?"
        elif nlp.vocab.strings[match_id] == "FAREWELL":
            return "Goodbye! Have a great day!"
    return None

# ============================
# Sentiment Analysis Function
# ============================
# Define sample intents
intent_phrases = {
    "greeting": ["hello", "hi", "hey", "howdy", "greetings"],
    "farewell": ["bye", "goodbye", "see you", "farewell"],
    "help": ["help", "assist", "support"],
}

def recognize_intent(text):
    doc = nlp(text.lower())
    for intent, phrases in intent_phrases.items():
        if any(phrase in text for phrase in phrases):
            return intent
    # Recognize greeting intents
    if any(greeting in text for greeting in ["hello", "hi", "hey", "how are you", "good morning", "good evening"]):
        return "greeting"
    elif "bye" in text or "goodbye" in text:
        return "farewell"
    elif "who are you" in text:
        return "identity"
    elif "help" in text or "assist" in text:
        return "help"
    return "unknown"


# ============================
# Initialize user modes and greeting tracking
# ============================
user_modes = {}
user_greeted = set()

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

# Parse user input using spaCy
def parse_command(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract named entities
    tokens = [(token.text, token.pos_) for token in doc]  # Tokenize text
    return entities, tokens

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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Analyze sentiment
    sentiment = analyze_sentiment(message.content)
    
    if sentiment == "positive":
        await message.channel.send("I'm glad you're feeling positive!")
    elif sentiment == "negative":
        await message.channel.send("I'm here for you. Let me know if there's anything I can help with.")

    await bot.process_commands(message)

# ============================
# Intent phrases
# ============================
intent_phrases = {
    "greeting": ["hello", "hi", "hey", "howdy", "greetings"],
    "farewell": ["bye", "goodbye", "see you", "farewell"],
    "help": ["help", "assist", "support"],
    "identity": ["who are you", "what are you"],
}

# ============================
# Recognize intent based on phrases
# ============================
def recognize_intent(text):
    text = text.lower()
    for intent, phrases in intent_phrases.items():
        if any(phrase in text for phrase in phrases):
            return intent
    return "unknown"

# ============================
# Analyze sentiment using TextBlob
# ============================
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "positive"
    elif analysis.sentiment.polarity < -0.1:
        return "negative"
    return "neutral"

# ============================
# Toggle between chat and command mode
# ============================
@bot.command(name='chatmode')
async def toggle_chatmode(ctx):
    user_id = ctx.author.id
    if user_modes.get(user_id) == "command":
        user_modes[user_id] = "chat"
        await ctx.send("Switched to chat mode! Feel free to talk to me naturally.")
    else:
        user_modes[user_id] = "command"
        await ctx.send("Switched to command-only mode. Use !commands to see available commands.")
        
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore bot's own messages

    user_id = message.author.id
    user_mode = user_modes.get(user_id, "command")  # Default to command mode if not set

    # If in command-only mode, process only commands
    if user_mode == "command":
        if message.content.startswith('!'):
            await bot.process_commands(message)
        return

    # Chat mode: respond to natural language queries
    if user_mode == "chat":
        # Recognize intents
        intent = recognize_intent(message.content)

        if intent == "greeting":
            await message.channel.send("Hello! How can I help you today?")
        elif intent == "farewell":
            await message.channel.send("Goodbye! Let me know if you need anything else.")
        elif intent == "help":
            await message.channel.send("I'm here to assist! Just ask me anything.")
        elif intent == "who":
            await message.channel.send("I'm SAPIENTBOT, here to chat and help you with whatever I can!")
        else:
            # Use OpenAI or similar for open-ended questions
            response = await generate_ai_response(message.content)
            await message.channel.send(response)

    # Ensure all bot commands are processed
    await bot.process_commands(message)

# ============================
# Handle all messages
# ============================
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore the bot's own messages

    user_id = message.author.id
    user_mode = user_modes.get(user_id, "command")  # Default to command mode

    # Process commands if in command mode
    if user_mode == "command" and message.content.startswith('!'):
        await bot.process_commands(message)
        return

    # Chat mode: interpret natural language queries
    if user_mode == "chat":
        intent = recognize_intent(message.content)

        if intent == "greeting":
            await message.channel.send("Hello! How can I assist you today?")
        elif intent == "farewell":
            await message.channel.send("Goodbye! Feel free to reach out anytime.")
        elif intent == "identity":
            await message.channel.send("I'm SAPIENTBOT, here to chat and help you with whatever I can!")
        elif intent == "help":
            await message.channel.send("I'm here to assist! Just ask me anything.")
        else:
            # Sentiment analysis and free-form AI response for unknown intent
            sentiment = analyze_sentiment(message.content)
            if sentiment == "positive":
                await message.channel.send("You seem to be in a happy mood! How can I assist you today?")
            elif sentiment == "negative":
                await message.channel.send("It seems like something is bothering you. How can I help?")
            else:
                response = await generate_ai_response(message.content)
                await message.channel.send(response)

    # Ensure all other bot commands are processed
    await bot.process_commands(message)

# ============================
# '!' COMMANDS & GROUPS / Gamification System
# ============================
class GamificationSystem:
    def __init__(self):
        self.user_rewards = {}  # Stores user rewards (points)
        self.user_levels = {}   # Stores user levels
        self.badges = {}        # Stores badges earned by users
        self.last_reward_time = {}  # Track the last time each user was rewarded

    def reward_user(self, user_id, points):
        # Ensure that new users are initialized properly
        if user_id not in self.user_rewards:
            self.user_rewards[user_id] = 0
            self.user_levels[user_id] = 1
            self.badges[user_id] = []
            self.last_reward_time[user_id] = datetime.now() - timedelta(days=1)  # Allow first bonus

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

        if new_level > self.user_levels[user_id]:
            self.user_levels[user_id] = new_level
            return True
        return False

    def assign_badges(self, user_id):
        points = self.user_rewards[user_id]
        if points >= 500 and '500 Club' not in self.badges[user_id]:
            self.badges[user_id].append('500 Club')
    
    def get_badges(self, user_id):
        return self.badges.get(user_id, [])

    def daily_bonus(self, user_id):
        now = datetime.now()
        last_reward = self.last_reward_time.get(user_id, now - timedelta(days=1))

        if (now - last_reward).days >= 1:
            self.user_rewards[user_id] += 50  # 50 bonus points
            self.last_reward_time[user_id] = now
            return True
        return False

gamification = GamificationSystem()

# ============================
# Fun Group & Commands Category
# ============================
@bot.group(name='fun', invoke_without_command=True)
async def fun(ctx):
    """🎉 Fun commands to lighten up your day!"""
    await ctx.send_help(ctx.command) 

# 1. Magic 8-Ball
@bot.command(name='8ball')
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

# 2. Coin Flip
@bot.command(name='flip')
async def coin_flip(ctx):
    outcome = random.choice(['Heads', 'Tails'])
    await ctx.send(f"🪙 The coin landed on: **{outcome}**!")

# 3. Dice Roll
@bot.command(name='roll')
async def roll_dice(ctx, dice: str):
    try:
        rolls, sides = map(int, dice.split('d'))
        results = [random.randint(1, sides) for _ in range(rolls)]
        await ctx.send(f"🎲 You rolled: {results} (Total: {sum(results)})")
    except ValueError:
        await ctx.send("Please use the format XdY (e.g., 2d6 for rolling two 6-sided dice).")

# 6. Trivia Game
@bot.command(name='trivia')
async def trivia(ctx):
    url = "https://opentdb.com/api.php?amount=1&type=multiple"
    try:
        response = requests.get(url).json()
    except requests.RequestException as e:
        await ctx.send("Couldn't fetch a trivia question at the moment. Please try again later.")
        return

    # Proceed with processing the response as before
    if response['response_code'] == 0:
        question = response['results'][0]['question']
        options = response['results'][0]['incorrect_answers']
        options.append(response['results'][0]['correct_answer'])
        random.shuffle(options)
        options_str = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)])

        await ctx.send(f"🧠 Trivia Time!\n{question}\n\n{options_str}\n\nType the number of your answer.")
        
        def check(m):
            return m.author == ctx.author and m.content.isdigit()

        try:
            answer = await bot.wait_for('message', check=check, timeout=15.0)
            if options[int(answer.content) - 1] == response['results'][0]['correct_answer']:
                await ctx.send("🎉 Correct! Great job.")
            else:
                await ctx.send(f"❌ Sorry, the correct answer was: {response['results'][0]['correct_answer']}")
        except:
            await ctx.send("⏰ Time's up!")
    else:
        await ctx.send("Couldn't fetch a trivia question. Try again later.")

# 7. Motivational Quotes
@bot.command(name='motivate')
async def motivate(ctx):
    quotes = [
        "Believe you can and you're halfway there.",
        "It always seems impossible until it’s done.",
        "You are stronger than you think.",
        "Keep going, you're doing amazing.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts."
    ]
    username = ctx.author.name
    await ctx.send(f"💪 {random.choice(quotes)} {username}, remember that you're capable of great things!")

# 8. Dad Joke Generator
@bot.command(name='dadjoke')
async def dad_joke(ctx):
    headers = {'Accept': 'application/json'}
    response = requests.get("https://icanhazdadjoke.com/", headers=headers).json()
    if 'joke' in response:
        await ctx.send(f"🤣 {response['joke']}")
    else:
        await ctx.send("Couldn't get a joke right now, try again later!")

# 9. Random Compliment
@bot.command(name='compliment')
async def compliment(ctx, *, member: discord.Member = None):
    if member is None:
        member = ctx.author

    compliments = [
        "You're an amazing person!", "You light up the room!", "Your smile is contagious.",
        "You have the best laugh!", "You bring out the best in other people."
    ]
    await ctx.send(f"{member.mention}, {random.choice(compliments)}! 😊")

# 10. Rock, Paper, Scissors Game
@bot.command(name='rps')
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
#11 Quiz
# Track when each user last used the !quiz command
user_last_quiz_time = {}
# Store active quizzes and their answers for each user
active_quizzes = {}
# Quiz cooldown time (1 day)
QUIZ_COOLDOWN = timedelta(days=1)
# Command to initiate a quiz
@bot.command(name='quiz')
async def quiz_command(ctx):
    user_id = ctx.author.id
    now = datetime.now()

    # Check if user has already taken the quiz today
    if user_id in user_last_quiz_time:
        last_quiz_time = user_last_quiz_time[user_id]
        if now - last_quiz_time < QUIZ_COOLDOWN:
            time_remaining = QUIZ_COOLDOWN - (now - last_quiz_time)
            await ctx.send(f"You've already taken the quiz today. Please try again in {time_remaining.seconds // 3600} hours.")
            return

    # Pick a random difficulty and question
    difficulty = random.choice(list(riddles.keys()))
    riddle = random.choice(riddles[difficulty])

    # Store the question and answer for the user
    active_quizzes[user_id] = {'question': riddle['question'], 'answer': riddle['answer'], 'time': now}

    # Send the quiz question to the user
    await ctx.send(f"Here's your quiz question: {riddle['question']}")
    await ctx.send("Reply with the correct answer using the `!quiz_answer` command!")

# Command to answer the quiz
@bot.command(name='quiz_answer')
async def quiz_answer_command(ctx, *, user_answer: str):
    user_id = ctx.author.id

    # Check if the user has an active quiz
    if user_id not in active_quizzes:
        await ctx.send("You don't have an active quiz. Use the `!quiz` command to get a question.")
        return

    # Get the stored quiz question and answer
    quiz_data = active_quizzes[user_id]
    correct_answer = quiz_data['answer'].lower()
    now = datetime.now()

    # Check if the answer is correct
    if user_answer.lower() == correct_answer:
        # Reward the user with 100 points
        gamification.reward_user(user_id, 100)
        total_points = gamification.get_user_rewards(user_id)

        await ctx.send(f"Correct! You've earned 100 points. Your total points: {total_points}")

        # Update the last quiz time for the user
        user_last_quiz_time[user_id] = now
    else:
        await ctx.send(f"Sorry, that's incorrect. The correct answer was: {correct_answer}")

    # Remove the active quiz for the user
    del active_quizzes[user_id]

#12 Riddles
# Riddles database
riddles = {
    'easy': [
        {"question": "What has to be broken before you can use it?", "answer": "egg"},
        {"question": "I’m tall when I’m young, and I’m short when I’m old. What am I?", "answer": "candle"},
        {"question": "What has legs but doesn’t walk?", "answer": "table"},
        {"question": "What is full of holes but still holds water?", "answer": "sponge"},
        {"question": "What runs, but never walks?", "answer": "water"},
        {"question": "What has ears but cannot hear?", "answer": "corn"}
    ],
    'intermediate': [
        {"question": "What month of the year has 28 days?", "answer": "all of them"},
        {"question": "What is full of holes but still holds water?", "answer": "sponge"},
        {"question": "What can travel around the world while staying in one spot?", "answer": "stamp"},
        {"question": "I have branches, but no fruit, trunk, or leaves. What am I?", "answer": "bank"},
        {"question": "If you drop me, I’m sure to crack, but smile at me and I’ll smile back. What am I?", "answer": "mirror"},
        {"question": "The more of this there is, the less you see. What is it?", "answer": "darkness"},
        {"question": "I shave every day, but my beard stays the same. What am I?", "answer": "barber"}
    ],
    'hard': [
        {"question": "What question can you never answer yes to?", "answer": "are you asleep"},
        {"question": "What gets wetter the more it dries?", "answer": "towel"},
        {"question": "What has a head, a tail, is brown, and has no legs?", "answer": "penny"},
        {"question": "A man is pushing his car along a road when he comes to a hotel. He shouts, 'I'm bankrupt!' Why?", "answer": "He's playing Monopoly."},
        {"question": "The eight of us go forth not back to protect our king from a foe’s attack. What are we?", "answer": "chess pawns"},
        {"question": "I am not alive, but I grow; I don’t have lungs, but I need air; I don’t have a mouth, but water kills me. What am I?", "answer": "fire"},
        {"question": "What flies without wings?", "answer": "time"}
    ],
    'professional': [
        {"question": "What can travel around the world while staying in the corner?", "answer": "stamp"},
        {"question": "The more of this there is, the less you see. What is it?", "answer": "darkness"},
        {"question": "What can bring back the dead, make you cry, make you laugh, make you young, is born in an instant, yet lasts a lifetime?", "answer": "memory"},
        {"question": "I’m not alive, but I can grow; I don’t have lungs, but I need air; I don’t have a mouth, and yet water kills me. What am I?", "answer": "fire"},
        {"question": "First you eat me, then you get eaten. What am I?", "answer": "fishhook"},
        {"question": "It cannot be seen, cannot be felt, cannot be heard, cannot be smelt. It lies behind stars and under hills, and empty holes it fills. It comes first and follows after, ends life, kills laughter. What is it?", "answer": "dark"},
        {"question": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", "answer": "echo"}
    ]
}

# Points for difficulty
points = {
    'easy': 5,
    'intermediate': 10,
    'hard': 15,
    'professional': 20
}

# Active riddles and users' answers
active_riddles = {}

# Command to give a riddle based on difficulty
@bot.command(name='riddle')
async def riddle_command(ctx, difficulty: str):
    user_name = ctx.author.display_name  # Get the user's display name

    # Select a random riddle from the specified difficulty level
    riddle = random.choice(riddles[difficulty])
    active_riddles[ctx.author.id] = {'question': riddle['question'], 'answer': riddle['answer'], 'difficulty': difficulty}
    
# Send the riddle to the user with their name
    await ctx.send(f"{user_name}, here's your {difficulty} riddle: {riddle['question']}")
    await ctx.send("Reply with the correct answer!")
    
# Command to check the user's answer
@bot.command(name='answer')
async def answer_command(ctx, *, user_answer: str):
    user_id = ctx.author.id

    if user_id not in active_riddles:
        await ctx.send("You don't have an active riddle. Use !riddle <difficulty> to get one.")
        return
    
    riddle_data = active_riddles[user_id]
    correct_answer = riddle_data['answer'].lower()
    difficulty = riddle_data['difficulty']

    if user_answer.lower() == correct_answer:
        # Reward the user based on the difficulty level
        reward_points = points[difficulty]
        gamification.reward_user(user_id, reward_points)
        total_points = gamification.get_user_rewards(user_id)

        await ctx.send(f"Correct! You've been rewarded {reward_points} points. Your total points: {total_points}")
    else:
        await ctx.send(f"Sorry, that's incorrect. The correct answer was: {correct_answer}")

    # Remove the active riddle for the user
    del active_riddles[user_id]

# Prevent command spam
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore the bot's own messages

    # Check for quiz spam
    if message.content.startswith('!quiz'):
        user_id = message.author.id
        now = datetime.now()
        if user_id in user_last_quiz_time:
            last_quiz_time = user_last_quiz_time[user_id]
            if now - last_quiz_time < QUIZ_COOLDOWN:
                await message.channel.send(f"You've already taken the quiz today. Please wait until tomorrow.")
                return

    # Process commands
    await bot.process_commands(message)

# ============================
# Spotify Group & Commands Category
# ============================
# Global variable to store the Spotify token
spotify_token = None
# Spotify OAuth setup for playback control
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state"
)
# Optimized Spotify token retrieval function
def get_spotify_token():
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        token_info = sp_oauth.get_access_token(as_dict=False)
    return token_info['access_token']

# Spotify Authentication for public commands (searching, top tracks)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Define the Spotify command group
@bot.group(name='spotify', invoke_without_command=True)
async def spotify(ctx):
    """🎶 Spotify commands for music lovers!"""
    await ctx.send_help(ctx.command)

# Define the authorize command within the spotify group
@spotify.command(name='authorize')
async def authorize_spotify(ctx):
    """🔗 Provides the Spotify authorization URL for first-time users."""
    auth_url = sp_oauth.get_authorize_url()
    await ctx.send(f"Please authorize the bot to access Spotify by visiting this link: {auth_url}")

# Define the play command within the spotify group
@spotify.command(name='play')
async def play_spotify(ctx):
    """▶️ Play Spotify!"""
    try:
        token = get_spotify_token()
        sp = spotipy.Spotify(auth=token)
        sp.start_playback()
        await ctx.send(f"{ctx.author.mention}, started Spotify playback. 🎶")
    except Exception as e:
        await ctx.send(f"Error starting playback: {e}")

# Define the pause command within the spotify group
@spotify.command(name='pause')
async def pause_spotify(ctx):
    """⏸️ Pause Spotify playback."""
    try:
        token = get_spotify_token()
        sp = spotipy.Spotify(auth=token)
        sp.pause_playback()
        await ctx.send(f"{ctx.author.mention}, paused Spotify playback. ⏸️")
    except Exception as e:
        await ctx.send(f"Error pausing playback: {e}")

# Define the current command within the spotify group
@spotify.command(name='current')
async def spotify_current(ctx):
    """🎵 Display the currently playing track on Spotify."""
    try:
        token = get_spotify_token()
        sp = spotipy.Spotify(auth=token)
        current_track = sp.current_playback()
        
        if current_track and current_track['is_playing']:
            track_name = current_track['item']['name']
            artist_name = current_track['item']['artists'][0]['name']
            await ctx.send(f"Now playing: '{track_name}' by {artist_name} 🎵")
        else:
            await ctx.send("No track is currently playing.")
    except Exception as e:
        await ctx.send(f"Error retrieving current playback: {e}")

# ============================
# Leaderboard Group & Commands Category
# ============================
# Define the Spotify command group
@bot.group(name='Score', invoke_without_command=True)
async def spotify(ctx):
    """ Check out your Score on the Leaderboard"""
    await ctx.send_help(ctx.command)

#Check User Points
@bot.command(name='points')
async def points_command(ctx):
    user_id = ctx.author.id
    total_points = gamification.get_user_rewards(user_id)
    await ctx.send(f"You have {total_points} points!")
    

# 1. !profile: Displays the user's profile, including total points, level, and badges.
@bot.command(name='profile')
async def profile_command(ctx):
    """Displays the user's profile and integrates RL action selection."""
    user_id = ctx.author.id
    total_points = gamification.get_user_rewards(user_id)
    level = gamification.get_user_level(user_id)
    badges = gamification.get_badges(user_id)
    badge_str = ", ".join(badges) if badges else "No badges yet."

    # Define the user's state and let the RL module choose an action
    user_state = f"user_{ctx.author.name}_profile"
    action = rl_module.choose_action(user_state)

    if action == 'reward_user':
        gamification.reward_user(user_id, 5)
        await ctx.send(f"Congratulations {ctx.author.name}, you've been rewarded 5 points! Total points: {total_points + 5}")
        rl_module.update_q_table(user_state, action, reward=1)  # Reward the bot for the successful action
    elif action == 'provide_tip':
        await ctx.send(f"Tip: Keep engaging with me to earn more points and rewards!")
        rl_module.update_q_table(user_state, action, reward=0.5)  # Partial reward for providing useful info
    else:
        await ctx.send(f"Here's your profile, {ctx.author.name}:\nTotal points: {total_points}, Level: {level}, Badges: {badge_str}")
        rl_module.update_q_table(user_state, action, reward=0)  # Neutral reward for no action


# ============================
# Voice Channel Group & Commands Category
# ============================
bot.group(name='Voice', invoke_without_command=True)
async def spotify(ctx):
    """ Let's hear your voice!"""
    await ctx.send_help(ctx.command)
# Command to join the voice channel
@bot.command(name='join')
async def join_voice(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel}")
    else:
        await ctx.send("You need to be in a voice channel to use this command.")

# Command to leave the voice channel
@bot.command(name='leave')
async def leave_voice(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected from the voice channel.")
    else:
        await ctx.send("I'm not in a voice channel!")
        


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Simple small talk phrases
    small_talk = {
        "who are you": "I'm SAPIENTBOT, here to help and chat with you!",
        "what's your purpose": "I'm designed to assist, answer questions, and make your experience enjoyable!",
        "tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field!"
    }

    if message.content.lower() in small_talk:
        await message.channel.send(small_talk[message.content.lower()])
    else:
        response = await generate_ai_response(message.content)
        await message.channel.send(response)

    await bot.process_commands(message)
    def update_user_context(user_id, topic):
        user_context[user_id] = topic

# Example function to retrieve the last conversation topic
def get_user_context(user_id):
    return user_context.get(user_id, None)

# ============================
# Error Handling
# ============================
# Handle errors
@bot.event
async def on_command_error(ctx, error):
    """Handles errors and provides feedback to the user."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Command not found. Type !commands to see the list of available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing arguments. Check the usage of the command.")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

# ============================
# Set up logging
# ============================
# Track user conversation topics
user_context = {}
def update_context(user_id, topic):
    user_context[user_id] = topic
def get_context(user_id):
    return user_context.get(user_id, "general")

logging.basicConfig(filename='sapientbot_response_times.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO)
def parse_command(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    tokens = [(token.text, token.pos_) for token in doc]
    
    logging.info(f"Entities: {entities}")
    logging.info(f"Tokens: {tokens}")
    
    return entities, tokens
# Dictionary to store response times for each command
command_times = {}
# Function to log response time
def log_response_time(command_name, response_time):
    # Log the response time for each command
    if command_name not in command_times:
        command_times[command_name] = []
    
    command_times[command_name].append(response_time)
    
    # Calculate average response time
    average_time = sum(command_times[command_name]) / len(command_times[command_name])
    
    # Log to the file
    logging.info(f"Command: {command_name} | Response Time: {response_time:.2f} seconds | Average: {average_time:.2f} seconds")

# Track user command history
user_command_history = {}
# Function to track user commands and provide context-aware responses
def track_user_command(ctx, command_name):
    user_id = ctx.author.id
    if user_id not in user_command_history:
        user_command_history[user_id] = []
    user_command_history[user_id].append(command_name)
    
# File to store user data
USER_DATA_FILE = 'user_data.json'

# ============================
# Function to generate response using OpenAI
# ============================
client = OPEN_API_ID
# Function to generate a response using OpenAI Chat API
async def generate_ai_response(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[{"role": "user", "content": message}],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"

# ============================
# Run Flask server in a separate process
# ============================
app = Flask(__name__)
def run_flask():
    app.run(port=8888)

flask_process = Process(target=run_flask)
flask_process.start()

# ============================
# Run the bot
# ============================
bot.run(DISCORD_TOKEN)
