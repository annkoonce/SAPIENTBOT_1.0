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
import matplotlib.pyplot as plt
import aiohttp
import tensorflow as tf

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
# LOAD ENVIRONMENT VARIABLES
# ============================
# Suppress TensorFlow and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
AI_KEY = os.getenv("AI_KEY")

# Check API keys
if not DISCORD_TOKEN or not AI_KEY:
    raise ValueError("DISCORD_TOKEN or AI_KEY is not set in the .env file.")

# Initialize OpenAI
openai.api_key = AI_KEY

# Example function to query OpenAI
async def ask_openai(question: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            max_tokens=150,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred while querying AI: {e}"

# ============================
# DATABASE CONNECTION
# ============================
def connect_db():
    """Establish a connection to the database."""
    conn = sqlite3.connect("sapientbot.db")
    return conn, conn.cursor()

# ============================
# HELPER FUNCTIONS
# ============================
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

user_context = {}
user_last_quiz_time = {}
QUIZ_COOLDOWN = timedelta(days=1)

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
# COMMAND DECORATORS
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
# BOT INITIALIZATION
# ============================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ============================
# MULTILINGUAL SUPPORT
# ============================
multilingual_greetings = {
    "en": ["hello", "hi", "hey"],
    "es": ["hola", "buenos días", "buenas tardes"],
    "fr": ["bonjour", "salut"],
    "de": ["hallo", "guten tag"],
    "pt": ["olá", "oi"]
}

multilingual_responses = {
    "greeting": {
        "en": "Hello! How can I assist you?",
        "es": "¡Hola! ¿Cómo puedo ayudarte?",
        "fr": "Bonjour! Comment puis-je vous aider?",
        "de": "Hallo! Wie kann ich Ihnen helfen?",
        "pt": "Olá! Como posso ajudá-lo?"
    }
}

# ============================
# GAMIFICATION SYSTEM
# ============================
class GamificationSystem:
    """Class to manage user rewards, levels, and badges."""

    def __init__(self):
        self.data = {"user_rewards": {}, "user_levels": {}}

    def reward_user(self, user_id, points):
        """Reward a user with points and handle level-up logic."""
        if user_id not in self.data["user_rewards"]:
            self.data["user_rewards"][user_id] = 0
            self.data["user_levels"][user_id] = 1

        self.data["user_rewards"][user_id] += points
        self.data["user_levels"][user_id] = self.data["user_rewards"][user_id] // 100

    def get_user_rewards(self, user_id):
        return self.data["user_rewards"].get(user_id, 0)

    def get_user_level(self, user_id):
        return self.data["user_levels"].get(user_id, 1)

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
async def leaderboard_command(ctx, page: int = 1):
    """
    Display the leaderboard with optional pagination.
    """
    users_per_page = 5
    start_index = (page - 1) * users_per_page
    end_index = start_index + users_per_page

    # Access user_rewards from the gamification.data dictionary
    user_rewards = gamification.data.get("user_rewards", {})
    top_users = sorted(user_rewards.items(), key=lambda x: x[1], reverse=True)
    total_pages = (len(top_users) + users_per_page - 1) // users_per_page

    if page < 1 or page > total_pages:
        await ctx.send(f"⚠️ Invalid page number. There are {total_pages} pages.")
        return

    leaderboard_text = f"**Leaderboard - Page {page}/{total_pages}**\n"
    for rank, (user_id, points) in enumerate(top_users[start_index:end_index], start=start_index + 1):
        try:
            user = await bot.fetch_user(int(user_id))  # Ensure user_id is cast to int
            leaderboard_text += f"{rank}. {user.name} - {points} points\n"
        except Exception:
            leaderboard_text += f"{rank}. [Unknown User] - {points} points\n"

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
    """
    Display available fun commands.
    """
    await ctx.send(
        "Check out the fun commands!\n\n"
        "💡 **Misc:**\n"
        "🎱 `!8ball` - Need answers from the Universe?\n"
        "🪙 `!flip` - Let's Flip a Coin!\n"
        "💪 `!motivate` - Need some Motivation?\n\n"
        "🧩 **Games:**\n"
        "  🧩 `!riddle` - Riddle me This\n"
        "  🎲 `!roll` - Roll D&D Dice\n"
        "  ✊✋✌️ `!rps` - Rock, Paper, Scissors\n"
        "  🧠 `!trivia`\n\n"
        "💬 **Chat:**\n"
        "  🗨️ `!ask` - Chat with context-aware conversations\n\n"
        "Type `!<command>` to use any of these!"
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
# Persistent session memory
session_memory = {}
def update_memory(user_id, message, memory_limit=5):
    """
    Update session memory for a user. Keeps the last `memory_limit` messages.
    """
    if user_id not in session_memory:
        session_memory[user_id] = []
    session_memory[user_id].append(message)
    if len(session_memory[user_id]) > memory_limit:
        session_memory[user_id].pop(0)
        
# 11. Translate command
@bot.command(name="translate")
async def translate_command(ctx, target_language: str, *, text: str):
    supported_languages = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'ja': 'Japanese', 'zh': 'Chinese', 'ko': 'Korean'
    }

    if target_language not in supported_languages:
        await ctx.send(f"Unsupported language code: {target_language}. Supported languages are: {', '.join(supported_languages.keys())}")
        return

    try:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        await ctx.send(f"Translated Text ({supported_languages[target_language]}): {translated_text}")
    except Exception as e:
        await ctx.send(f"Translation failed: {e}")

# ============================
# Feedback Loop
# ============================ 
# Feedback storage (simple JSON for demonstration purposes)
FEEDBACK_FILE = "feedback.json"

# Load feedback from JSON file
def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            else:
                return []
    except json.JSONDecodeError:
        return []

# Save feedback to JSON file
def save_feedback(feedback_data):
    with open(FEEDBACK_FILE, 'w') as file:
        json.dump(feedback_data, file, indent=4)

# Command for users to provide feedback
@bot.command(name='submit_feedback')
async def submit_feedback(ctx, rating: int, *, comments: str = None):
    if rating < 1 or rating > 5:
        await ctx.send("Please provide a rating between 1 and 5.")
        return

    # Load existing feedback
    feedback_data = load_feedback()

    # Check if user has already submitted feedback
    for entry in feedback_data:
        if entry["user_id"] == str(ctx.author.id):
            # Update existing feedback
            entry["rating"] = rating
            entry["comments"] = comments
            entry["timestamp"] = datetime.now().isoformat()
            break
    else:
        # Add new feedback if user has not submitted yet
        feedback_entry = {
            "rating": rating,
            "comments": comments,
            "username": ctx.author.name,
            "user_id": str(ctx.author.id),
            "timestamp": datetime.now().isoformat()
        }
        feedback_data.append(feedback_entry)

    # Save feedback
    save_feedback(feedback_data)

    await ctx.send("Your feedback has been recorded!")

# Command to visualize feedback metrics
@bot.command(name='view_metrics')
async def view_metrics(ctx):
    # Load feedback data
    feedback_data = load_feedback()
    if not feedback_data:
        await ctx.send("No feedback data available.")
        return

    # Extract ratings and calculate stats
    ratings = [entry["rating"] for entry in feedback_data]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    unique_users = len(set(entry["user_id"] for entry in feedback_data))

    # Plot the feedback ratings
    plt.figure(figsize=(8, 4))
    plt.hist(ratings, bins=range(1, 7), edgecolor='black', align='left')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title(f'User Feedback Ratings (Avg: {avg_rating:.2f}, Users: {unique_users})')
    plt.xticks(range(1, 6))

    # Save and send the plot
    plt.savefig('feedback_metrics.png')
    plt.close()

    with open('feedback_metrics.png', 'rb') as file:
        await ctx.send(file=discord.File(file, 'feedback_metrics.png'))
# ============================
# LOGGING
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sapientbot.log"),
        logging.StreamHandler()
    ]
)

# ============================
# RUN THE BOT
# ============================
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
