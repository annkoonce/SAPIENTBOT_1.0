import os
import discord
import spacy
from spacy.matcher import Matcher
from textblob import TextBlob
from discord.ext import commands
import random
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, request
import threading
import logging
import openai
from datetime import datetime, timedelta
import json
from logging.handlers import RotatingFileHandler

# Load environment variables and validate them
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')
SERP_API_ID = os.getenv('SERP_API_ID')
OPEN_API_ID = os.getenv('OPEN_API_ID')

required_env_vars = [DISCORD_TOKEN, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, SERP_API_ID, OPEN_API_ID]
for var, name in zip(required_env_vars, ['DISCORD_TOKEN', 'SPOTIPY_CLIENT_ID', 'SPOTIPY_CLIENT_SECRET', 'SPOTIPY_REDIRECT_URI', 'SERP_API_ID', 'OPEN_API_ID']):
    if not var:
        raise ValueError(f"Environment variable {name} is not set. Please add it to your .env file.")

# Load spaCy model for NLU
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Setup logging to write to a rotating file
log_handler = RotatingFileHandler('sapientbot_response_times.log', maxBytes=5*1024*1024, backupCount=5)
logging.basicConfig(handlers=[log_handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Track user command history
user_command_history = {}

# Setup function to track user commands and provide context-aware responses
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

    def setup_user(self, user_id):
        if user_id not in self.user_rewards:
            self.user_rewards[user_id] = 0
            self.user_levels[user_id] = 1
            self.badges[user_id] = []
            self.last_reward_time[user_id] = datetime.now() - timedelta(days=1)  # Allow first bonus

    def reward_user(self, user_id, points):
        self.setup_user(user_id)
        now = datetime.now()

        if now - self.last_reward_time[user_id] >= timedelta(hours=1):
            self.user_rewards[user_id] += points
            self.last_reward_time[user_id] = now
            return True
        return False

    def grant_bonus(self, user_id):
        self.setup_user(user_id)
        now = datetime.now()

        if now - self.last_reward_time[user_id] >= timedelta(days=1):
            self.user_rewards[user_id] += 50  # 50 bonus points
            self.last_reward_time[user_id] = now
            return True
        return False

# Create instance of GamificationSystem
gamification = GamificationSystem()

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

# Create bot object
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
user_greeted = set()

# List of custom greetings for specific users
custom_greetings = {
    'ColorsAndLights': 'Ara ara, ColorsAndLights-kun~',
    'kilanaann': 'Hello Mother-unit!',
    'itsbiskitty': 'I have been trying to contact you about your card extended warranty.'
    # Add more custom greetings here for other users
}

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
        


@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Ignore bot's own messages

    # Check for custom greetings
    if message.author.name in custom_greetings:
        await message.channel.send(custom_greetings[message.author.name])

    # Process commands
    await bot.process_commands(message)

# Flask app threading to run alongside the bot
def run_flask():
    app.run(host='0.0.0.0', port=5000)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Main bot execution
bot.run(DISCORD_TOKEN)
