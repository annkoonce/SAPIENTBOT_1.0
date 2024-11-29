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

from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from discord.ext import commands
from langdetect import detect
from deep_translator import GoogleTranslator
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

# ============================
# Database Connection
# ============================
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

# Initialize AI-powered intent detection model
intent_classifier = pipeline("text-classification", model="bert-base-multilingual-cased")

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

# Function to detect intent using AI
def detect_intent_with_ai(message_content):
    try:
        result = intent_classifier(message_content)
        intent = result[0]["label"]
        return intent
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return "UNKNOWN"

# Function to generate AI-based responses dynamically
def generate_ai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI response generation failed: {e}")
        return "I'm sorry, I couldn't process your request."

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

# Function to detect language and respond based on greetings

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

# Detect and respond to greetings
def detect_language_and_respond(message_content):
    detected_lang = detect_language(message_content)
    for lang, greetings in multilingual_greetings.items():
        if any(greet in message_content.lower() for greet in greetings):
            return multilingual_responses.get(lang, multilingual_responses["en"])
    return None

# Expanded greetings list for each language
multilingual_greetings = {
    "en": ["hello", "hi", "hey", "howdy", "greetings"],
    "es": ["hola", "buenos días", "buenas tardes", "buenas noches"],  # Spanish
    "fr": ["bonjour", "salut", "bonsoir"],  # French
    "pt": ["olá", "oi", "bom dia", "boa tarde", "boa noite"],  # Portuguese
    "de": ["hallo", "guten tag", "guten morgen", "guten abend", "hi"],  # German
    "tl": ["kumusta", "magandang umaga", "magandang hapon", "magandang gabi", "hello", "hi"]  # Tagalog
}
# Multilingual responses for greetings
multilingual_responses = {
    "en": "Hello! How can I assist you?",
    "es": "¡Hola! ¿Cómo puedo ayudarte?",
    "fr": "Bonjour! Comment puis-je vous aider?",
    "pt": "Olá! Como posso ajudá-lo?",
    "de": "Hallo! Wie kann ich Ihnen helfen?",  
    "tl": "Kumusta! Paano kita matutulungan?"  
}

# Training data for multilingual NLP
TRAINING_DATA = {
    "en": [
        ("Hello", {"intent": "greeting"}),
        ("How can you assist?", {"intent": "help"})
    ],
    "es": [
        ("Hola", {"intent": "greeting"}),
        ("¿Cómo puedes ayudarme?", {"intent": "help"})
    ],
    "tl": [
        ("Kumusta", {"intent": "greeting"}),
        ("Paano kita matutulungan?", {"intent": "help"})
    ],
    # Add more training data for other languages...
}

# Feedback function
def save_multilingual_feedback(user_id, lang, query, response, feedback):
    user_feedback.append({
        "user_id": user_id,
        "lang": lang,
        "query": query,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    })

multilingual_intents = {
    "smalltalk": {
        "en": ["how are you", "what’s up", "how’s it going", "what’s going on"],
        "es": ["cómo estás", "qué tal", "cómo te va"],
        "fr": ["comment ça va", "ça va", "comment vas-tu"],
        "pt": ["como está", "como vai", "tudo bem"],
        "de": ["wie geht's", "wie läuft's", "alles gut"],
        "tl": ["kamusta ka", "ano na", "anong balita"],
    },
    "farewell": {
        "en": ["goodbye", "bye", "see you", "farewell"],
        "es": ["adiós", "hasta luego", "nos vemos"],
        "fr": ["au revoir", "à bientôt", "adieu"],
        "pt": ["adeus", "tchau", "até logo"],
        "de": ["tschüss", "auf wiedersehen", "bis bald"],
        "tl": ["paalam", "hanggang sa muli", "bye"],
    },
    "help": {
        "en": ["help", "assist me", "i need help", "how do i"],
        "es": ["ayuda", "ayúdame", "necesito ayuda"],
        "fr": ["aidez-moi", "j'ai besoin d'aide", "comment faire"],
        "pt": ["ajude-me", "preciso de ajuda", "como faço"],
        "de": ["hilfe", "hilf mir", "wie mache ich"],
        "tl": ["tulong", "kailangan ko ng tulong", "paano ba"],
    },
}

multilingual_responses.update({
    "smalltalk": {
        "en": "I'm just a bot, but I'm here to help you!",
        "es": "¡Soy solo un bot, pero estoy aquí para ayudarte!",
        "fr": "Je suis juste un bot, mais je suis là pour vous aider!",
        "pt": "Sou apenas um bot, mas estou aqui para ajudá-lo!",
        "de": "Ich bin nur ein Bot, aber ich bin hier, um dir zu helfen!",
        "tl": "Isa lang akong bot, pero nandito ako para tumulong sa'yo!",
    },
    "farewell": {
        "en": "Goodbye! Take care!",
        "es": "¡Adiós! ¡Cuídate!",
        "fr": "Au revoir! Prenez soin de vous!",
        "pt": "Adeus! Cuide-se!",
        "de": "Tschüss! Pass auf dich auf!",
        "tl": "Paalam! Mag-ingat ka!",
    },
    "help": {
        "en": "I'm here to assist! What do you need help with?",
        "es": "¡Estoy aquí para ayudar! ¿Con qué necesitas ayuda?",
        "fr": "Je suis ici pour vous aider! Avec quoi avez-vous besoin d'aide?",
        "pt": "Estou aqui para ajudar! Com o que você precisa de ajuda?",
        "de": "Ich bin hier, um zu helfen! Wobei brauchst du Hilfe?",
        "tl": "Nandito ako para tumulong! Ano ang kailangan mong tulong?",
    },
})

def detect_intent_and_respond(message_content):
    try:
        detected_lang = detect(message_content)
    except:
        detected_lang = "en"  # Default to English if detection fails

    for intent, phrases in multilingual_intents.items():
        if detected_lang in phrases and any(phrase in message_content.lower() for phrase in phrases[detected_lang]):
            response = multilingual_responses[intent].get(detected_lang, multilingual_responses[intent]["en"])
            return response

    return None

# Event handler for messages
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Process commands first
    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    # Step 1: Detect greetings and respond
    response = detect_language_and_respond(message.content)
    if response:
        await message.channel.send(response)
        return

    # Step 2: Detect intent using AI
    intent = detect_intent_with_ai(message.content)
    if intent == "GREETING":
        await message.channel.send("Hello! How can I assist you?")
    elif intent == "HELP":
        await message.channel.send("I'm here to help! What do you need assistance with?")
    elif intent == "FAREWELL":
        await message.channel.send("Goodbye! Let me know if you need anything else.")
    else:
        # Step 3: Generate AI-based dynamic response
        ai_response = generate_ai_response(message.content)
        await message.channel.send(ai_response)

    # Process remaining commands
    await bot.process_commands(message)

# Feedback command
@bot.command(name="feedback")
async def feedback_command(ctx, feedback: int, *, query: str):
    detected_lang = detect(query)
    response = f"Thank you for your feedback: {feedback}/5"
    save_multilingual_feedback(ctx.author.id, detected_lang, query, response, feedback)
    await ctx.send(response)

# Translate command
@bot.command(name="translate")
async def translate_command(ctx, target_language: str, *, text: str):
    try:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        await ctx.send(f"Translated Text ({target_language}): {translated_text}")
    except Exception as e:
        await ctx.send(f"Translation failed: {e}")
        
# ============================
# Database Connection
# ============================
# Load a pretrained Hugging Face pipeline for multilingual text classification
intent_classifier = pipeline("text-classification", model="bert-base-multilingual-cased")

def detect_intent_with_ai(message_content):
    # Use the model to classify the intent
    result = intent_classifier(message_content)
    intent = result[0]["label"]
    return intent

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
    if analysis.sentiment.polarity > 0.2:
        return "positive"
    elif analysis.sentiment.polarity < -0.2:
        return "negative"
    else:
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

gamification = GamificationSystem()
# ============================
# Score Commands
# ============================ 
@bot.group(name='score', invoke_without_command=True)
async def score(ctx):
    """Check your scores and progress!"""
    await ctx.send_help(ctx.command)

@score.command(name='points')
async def points_command(ctx):
    user_id = str(ctx.author.id)
    total_points = gamification.get_user_rewards(user_id)
    await ctx.send(f"You have {total_points} points!")

@score.command(name='leaderboard')
async def leaderboard_command(ctx):
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
        "  ask\n"
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

# 2. Dice Roll
@fun.command(name='roll')
async def roll_dice(ctx, dice: str):
    try:
        rolls, sides = map(int, dice.split('d'))
        results = [random.randint(1, sides) for _ in range(rolls)]
        await ctx.send(f"🎲 You rolled: {results} (Total: {sum(results)})")
    except ValueError:
        await ctx.send("Please use the format XdY (e.g., 2d6 for rolling two 6-sided dice).")

# 3. Trivia Game

@fun.command(name='trivia')
async def trivia(ctx):
    url = "https://opentdb.com/api.php?amount=1&type=multiple"
    try:
        response = requests.get(url).json()
    except requests.RequestException:
        await ctx.send("Couldn't fetch a trivia question at the moment. Please try again later.")
        return

    if response.get('response_code') == 0:
        question_data = response['results'][0]
        question = html.unescape(question_data['question'])
        correct_answer = html.unescape(question_data['correct_answer'])
        options = [html.unescape(opt) for opt in question_data['incorrect_answers']]
        options.append(correct_answer)
        random.shuffle(options)

        options_str = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)])
        await ctx.send(f"🧠 Trivia Time!\n{question}\n\n{options_str}\n\nType the number of your answer.")

        def check(m):
            return m.author == ctx.author and m.content.isdigit()

        try:
            answer = await bot.wait_for('message', check=check, timeout=30)
            selected_option = options[int(answer.content) - 1]
            if selected_option == correct_answer:
                points = 50
                user_id = str(ctx.author.id)
                gamification.reward_user(user_id, points)
                total_points = gamification.get_user_rewards(user_id)
                await ctx.send(f"🎉 Correct! You've earned {points} points. Total Points: {total_points}")
            else:
                await ctx.send(f"❌ Incorrect! The correct answer was: {correct_answer}")
        except (IndexError, ValueError):
            await ctx.send("Invalid response. Please type the number corresponding to your answer.")
        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")
    else:
        await ctx.send("Couldn't fetch a trivia question. Try again later.")
        
# 4. Dad Joke Generator
@fun.command(name='dadjoke')
async def dad_joke(ctx):
    headers = {'Accept': 'application/json'}
    response = requests.get("https://icanhazdadjoke.com/", headers=headers).json()
    if 'joke' in response:
        await ctx.send(f"🤣 {response['joke']}")
    else:
        await ctx.send("Couldn't get a joke right now, try again later!")

# 5. Random Compliment
@fun.command(name='compliment')
async def compliment(ctx, *, member: discord.Member = None):
    if member is None:
        member = ctx.author
    compliments = [
        "You're an amazing person!", "You light up the room!", "Your smile is contagious.",
        "You have the best laugh!", "You bring out the best in other people."
    ]
    await ctx.send(f"{member.mention}, {random.choice(compliments)}! 😊")

# 6. Rock, Paper, Scissors Game
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

# 7. Quiz Game (Example of an additional command)
@fun.command(name='quiz')
async def quiz_command(ctx):
    # Quiz logic here (similar to previous code)
    await ctx.send("Quiz functionality is under the Fun group now!")
    
# 8.Add Coin Flip to Fun group
@fun.command(name='flip')
async def coin_flip(ctx):
    outcome = random.choice(['Heads', 'Tails'])
    await ctx.send(f"🪙 The coin landed on: **{outcome}**!")

# 9. Add Motivational Quotes to Fun group
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
    
# 10. Riddle and Riddle Commands database
@fun.command(name='riddle')
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
    
# Select a random riddle from the specified difficulty level
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

# 11. Register a command to ask OpenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to query Google Custom Search API
def google_search(query, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        return search_results.get("items", [])
    except Exception as e:
        logging.error(f"Error occurred while searching: {e}")
        return []

# Function to handle user queries, integrating search if confidence is low
def handle_user_query(query: str) -> str:
    doc = nlp(query)
    
    # Assuming low-confidence detection is based on entity recognition or similar checks
    if len(doc.ents) == 0:  # No entities found, indicating ambiguity
        # Fallback to Google search
        search_results = google_search(query)
        if search_results:
            response = "I found some information online that might help:\n"
            for result in search_results:
                response += f"- {result['title']}: {result['link']}\n"
            return response
        else:
            return "I'm sorry, I couldn't find enough information on that topic."
    else:
        # Handle confidently recognized queries normally
        return f"Here's what I understand about '{query}'..."

# Example usage of concurrent querying
def concurrent_search(queries):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(google_search, queries))
    return results

# Function to query OpenAI's GPT model
async def ask_openai(question: str) -> str:
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # You can still use GPT-4 if you have access
            prompt=question,
            max_tokens=150,
            temperature=0.7,  # Adjust temperature for more or less creativity
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error occurred while querying OpenAI: {e}")
        return f"Error: {str(e)}"
@fun.command(name='ask')
async def ask_question(ctx, *, question: str):
    # First try OpenAI
    await ctx.send("Let me think...")
    response_openai = await ask_openai(question)

    if response_openai and "Error" not in response_openai:
        main_answer = response_openai
    else:
        main_answer = "Please see search links below."

    # Search for related articles/links based on the question using Google search
    related_links = []
    try:
        # Correct usage of the googlesearch `search()` function
        for link in search(question, stop=3):  # The `stop` parameter limits the number of results to 3
            related_links.append(link)
    except Exception as e:
        related_links = [f"Error fetching related links: {str(e)}"]

    # Build the final response
    response = f"**OpenAI's Response:**\n{main_answer}\n\n**Here are some related links for further reading:**\n"
    for idx, link in enumerate(related_links, 1):
        response += f"{idx}. {link}\n"

    # Send the response back to the user
    await ctx.send(response)

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