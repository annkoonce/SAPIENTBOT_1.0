import os
import discord
import spacy
from spacy.matcher import Matcher
from textblob import TextBlob
from discord.ext import commands
import random
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from flask import Flask, request
import threading
import webbrowser
import logging
import openai
import serpapi 
from serpapi import GoogleSearch
from datetime import datetime, timedelta
import unittest
from unittest.mock import MagicMock
from googlesearch import search
import time
import logging
import SB_usertracking

# Load environment variables
load_dotenv()

# Load spaCy model for NLU
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Setup logging to write to a file
logging.basicConfig(filename='sapientbot_response_times.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Track user command history
user_command_history = {}
# Function to track user commands and provide context-aware responses
def track_user_command(ctx, command_name):
    user_id = ctx.author.id
    if user_id not in user_command_history:
        user_command_history[user_id] = []
    user_command_history[user_id].append(command_name)

# Gamification System
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

# Create instances of RLModule and GamificationSystem
gamification = GamificationSystem()

# Riddles database
riddles = {
    'easy': [
        {"question": "What has to be broken before you can use it?", "answer": "egg"},
        {"question": "I’m tall when I’m young, and I’m short when I’m old. What am I?", "answer": "candle"}
    ],
    'intermediate': [
        {"question": "What month of the year has 28 days?", "answer": "all of them"},
        {"question": "What is full of holes but still holds water?", "answer": "sponge"}
    ],
    'hard': [
        {"question": "What question can you never answer yes to?", "answer": "are you asleep"},
        {"question": "What gets wetter the more it dries?", "answer": "towel"}
    ],
    'professional': [
        {"question": "What can travel around the world while staying in the corner?", "answer": "stamp"},
        {"question": "The more of this there is, the less you see. What is it?", "answer": "darkness"}
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

# Bot Token and API keys
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')
SERP_API_ID = os.getenv('SERP_API_ID')
OPEN_API_ID = os.getenv('OPEN_API_ID')

# Flask app for OAuth callback
app = Flask(__name__)

# Global variable to store the Spotify token
spotify_token = None

# Spotify OAuth setup for playback control
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state"
)

# Define patterns for greetings and farewells
greetings = [{"LOWER": "hello"}, {"LOWER": "hi"}, {"LOWER": "hey"}]
farewells = [{"LOWER": "bye"}, {"LOWER": "goodbye"}, {"LOWER": "see"}, {"LOWER": "you"}]
matcher.add("GREETING", [greetings])
matcher.add("FAREWELL", [farewells])

# Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Command to give a riddle based on difficulty
@bot.command(name='riddle')
async def riddle_command(ctx, difficulty: str):
    # Track user interaction
    if difficulty not in riddles:
        await ctx.send(f"Invalid difficulty level. Choose from: {', '.join(riddles.keys())}")
        return
    
    # Select a random riddle from the specified difficulty level
    riddle = random.choice(riddles[difficulty])
    active_riddles[ctx.author.id] = {'question': riddle['question'], 'answer': riddle['answer'], 'difficulty': difficulty}
    
    # Send the riddle to the user
    await ctx.send(f"Here's your {difficulty} riddle: {riddle['question']}")
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

# Command to check the user's points
@bot.command(name='points')
async def points_command(ctx):
    user_id = ctx.author.id
    total_points = gamification.get_user_rewards(user_id)
    await ctx.send(f"You have {total_points} points!")

# Sentiment Analysis Function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

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

# Dictionary of common greetings and responses
greetings_responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! How’s it going?",
    "good morning": "Good morning! Hope you're having a great day.",
    "good afternoon": "Good afternoon! How may I help you?",
    "good evening": "Good evening! What can I assist you with?",
    "howdy": "Howdy! What’s up?",
    "greetings": "Greetings! How can I be of service?",
    "what's up": "Not much, you?",
    "how's it going": "I’m good, thanks! How about you?",
    "nice to meet you": "Nice to meet you too!",
    "pleased to meet you": "Pleased to meet you too!",
    "good to see you": "Good to see you too!",
    "yo": "Yo! What’s up?",
    "hiya": "Hiya! What’s new?",
    "how are you": "I'm fine, thank you! How about you?",
    "welcome": "Thank you! What can I help you with?",
    "how do you do": "I'm doing well, how about you?"
}

# Handle Natural Conversation with Sentiment Awareness and context-based responses# Command to handle natural conversation with sentiment awareness and greeting responses
@bot.command(name='talk')
async def conversation_handler(ctx, *, message: str):
    # Analyze sentiment as before
    sentiment = analyze_sentiment(message)
    
    # Normalize the message to lowercase for matching
    normalized_message = message.lower()

    # Check if the message contains any of the predefined greetings
    response = None
    for greeting in greetings_responses:
        if greeting in normalized_message:
            response = greetings_responses[greeting]
            break
    
    if response:
        # If a greeting is detected, respond with the mapped response
        await ctx.send(response)
    else:
        # If no greeting is detected, respond based on the sentiment
        if sentiment == "positive":
            await ctx.send("You seem to be in a happy mood! How can I assist you today?")
        elif sentiment == "negative":
            await ctx.send("It seems like something is bothering you. How can I help?")
        else:
            await ctx.send("I'm here to assist you. What can I do for you today?")

# Unit Tests for sentiment analysis and other features
class TestBotFunctions(unittest.TestCase):
    def test_analyze_sentiment_positive(self):
        text = "I am feeling great!"
        result = analyze_sentiment(text)
        self.assertEqual(result, "positive")

    def test_analyze_sentiment_negative(self):
        text = "I am feeling sad."
        result = analyze_sentiment(text)
        self.assertEqual(result, "negative")

    def test_analyze_sentiment_neutral(self):
        text = "It is an average day."
        result = analyze_sentiment(text)
        self.assertEqual(result, "neutral")

# Spotify Authentication for public commands (searching, top tracks)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Reinforcement Learning Module
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

# Spotify OAuth route
@app.route('/callback')
def callback():
    global spotify_token
    token_info = sp_oauth.get_access_token(request.args['code'])
    spotify_token = token_info['access_token']
    return "Spotify Authorization successful! You can close this tab and return to Discord."

### Spotify Commands
# 1. !spotify_current: Displays the current track playing on Spotify.
@bot.command(name='spotify_current')
async def spotify_current_track(ctx):
    """Displays the current playing track on Spotify."""
    results = sp.current_playback()
    if results and results['is_playing']:
        track = results['item']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        await ctx.send(f"Currently playing: {track_name} by {artist_name}")
    else:
        await ctx.send("Nothing is currently playing on Spotify.")

# 2. !spotify_search <track name>: Searches for a track on Spotify and returns the top result.
@bot.command(name='spotify_search')
async def spotify_search(ctx, *, query: str):
    """Searches for a track on Spotify and returns the top result."""
    entities, tokens = parse_command(query)  # Use spaCy to process the input
    results = sp.search(q=query, limit=1, type='track')
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        track_url = track['external_urls']['spotify']
        await ctx.send(f"Top result: {track_name} by {artist_name}\n{track_url}")
    else:
        await ctx.send("No results found.")
        

# 3. !spotify_top_tracks <artist name>: Displays the top tracks for a given artist.
@bot.command(name='spotify_top_tracks')
async def spotify_top_tracks(ctx, *, artist_name: str):
    """Displays the top tracks of a given artist."""
    results = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
    if results['artists']['items']:
        artist = results['artists']['items'][0]
        artist_id = artist['id']
        top_tracks = sp.artist_top_tracks(artist_id)['tracks']
        
        if top_tracks:
            message = f"Top tracks for {artist['name']}:\n"
            for i, track in enumerate(top_tracks[:5]):
                track_name = track['name']
                track_url = track['external_urls']['spotify']
                message += f"{i + 1}. {track_name} - {track_url}\n"
            await ctx.send(message)
        else:
            await ctx.send(f"No top tracks found for {artist_name}.")
    else:
        await ctx.send(f"Artist {artist_name} not found.")

# 4. !authorize_spotify: Initiates the Spotify OAuth authorization process.
@bot.command(name='authorize_spotify')
async def authorize_spotify(ctx):
    """Initiates Spotify OAuth authorization."""
    auth_url = sp_oauth.get_authorize_url()
    webbrowser.open(auth_url)
    await ctx.send(f"Please authorize Spotify by clicking on this link: {auth_url}")

# 5. !spotify_play: Starts playback on an active Spotify device (after authorization).
@bot.command(name='spotify_play')
async def spotify_play(ctx):
    """Starts playback on an active Spotify device."""
    if spotify_token is None:
        await ctx.send("You need to authorize Spotify first using !authorize_spotify.")
        return
    sp_user = spotipy.Spotify(auth=spotify_token)
    devices = sp_user.devices()
    if not devices['devices']:
        await ctx.send("No active Spotify devices found.")
    else:
        sp_user.start_playback()
        await ctx.send("Playback started!")

# Check if 'ask' command is already registered
existing_command = bot.get_command('ask')

# If it exists, remove it before registering the new one
if existing_command:
    bot.remove_command('ask')

# Function to ask OpenAI
def ask_openai(question):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

# Function to ask SerpApi
def ask_serpapi(query):
    try:
        params = {
            "q": query,
            "api_key": os.getenv('SERP_API_ID'),
            "num": 1  # Limit to one result
        }
        search = GoogleSearch(params)
        result = search.get_dict()

        # Extract snippet or result from the response
        snippet = result.get('organic_results', [{}])[0].get('snippet', 'No answer found.')
        return snippet
    except Exception as e:
        return f"Error with SerpApi: {str(e)}"

# Register a command to ask OpenAI and SerpApi
@bot.command(name='ask')
async def ask_question(ctx, *, question: str):
    # First try OpenAI, fallback to SerpApi if needed
    response_openai = ask_openai(question)
    
    if response_openai and "Error" not in response_openai:
        main_answer = response_openai
    else:
        main_answer = ask_serpapi(question)

    # Search for related articles/links based on the question using Google search
    related_links = []
    for link in search(question, num_results=3):  # You can adjust the number of results
        related_links.append(link)

    # Build the final response
    response = f"{main_answer}\n\nHere are some related links for further reading:\n"
    for idx, link in enumerate(related_links, 1):
        response += f"{idx}. {link}\n"
    
    # Send the response back to the user
    await ctx.send(response)

### Gamification Commands
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
        await ctx.send(f"Tip: Keep engaging with the bot to earn more points and rewards!")
        rl_module.update_q_table(user_state, action, reward=0.5)  # Partial reward for providing useful info
    else:
        await ctx.send(f"Here's your profile, {ctx.author.name}:\nTotal points: {total_points}, Level: {level}, Badges: {badge_str}")
        rl_module.update_q_table(user_state, action, reward=0)  # Neutral reward for no action

@bot.command(name='leaderboard')
async def leaderboard_command(ctx):
    """Displays the top 5 users with the most points, and uses RL to randomly reward users."""
    # Sort the users by points in descending order
    leaderboard = sorted(gamification.user_rewards.items(), key=lambda item: item[1], reverse=True)[:5]

    if not leaderboard:
        await ctx.send("No rewards data yet.")
        return

    # Create the leaderboard message
    message = "**🏆 Leaderboard 🏆**\n"
    for rank, (user_id, points) in enumerate(leaderboard, 1):
        user = await bot.fetch_user(user_id)  # Fetch the Discord user object
        if user:  # Check if user was found
            username = user.name  # Get the username from the user object
        else:
            username = str(user_id)  # Fallback to user ID if username can't be fetched

        message += f"{rank}. {username} - {points} points\n"

        # Apply RL action to possibly reward the leaderboard users
        user_state = f"leaderboard_{username}"
        action = rl_module.choose_action(user_state)
        if action == 'reward_user':
            gamification.reward_user(user_id, 5)
            message += f"{username} received a 5-point reward!\n"
        elif action == 'provide_tip':
            message += f"{username}, keep up the good work to stay on top!\n"

    await ctx.send(message)

# Handle @mention or DM interactions
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore the bot's own messages

    if isinstance(message.channel, discord.DMChannel):
        # Respond to messages directly in private messages
        response = handle_greetings(message.content)
        if response:
            await message.channel.send(response)
        else:
            sentiment = analyze_sentiment(message.content)
            if sentiment == "positive":
                await message.channel.send("I'm glad you're feeling great! How can I assist you?")
            elif sentiment == "negative":
                await message.channel.send("I'm sorry to hear that. Is there anything I can do to help?")
            else:
                await message.channel.send("How can I assist you today?")
    elif bot.user.mentioned_in(message):
        # Respond to messages if @mentioned in a server
        response = handle_greetings(message.content)
        if response:
            await message.channel.send(response)
        else:
            await message.channel.send("Hello! You mentioned me? How can I assist you today?")

    await bot.process_commands(message)

# Error Handling
@bot.event
async def on_command_error(ctx, error):
    """Handles errors and provides feedback to the user."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Command not found. Type !commands to see the list of available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing arguments. Check the usage of the command.")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

# Run Flask server for Spotify OAuth in a background thread
def run_flask():
    app.run(port=8888)

# Run Flask server in a background thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Set up logging
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

# Run the bot
bot.run(DISCORD_TOKEN)