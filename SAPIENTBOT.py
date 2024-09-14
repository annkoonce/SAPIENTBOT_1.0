import os
import discord
import random
import numpy as np
import requests
from discord.ext import commands
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# Bot Token and API keys
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Spotify Setup
sp = Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Discord Bot Setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)

# Q-table for reinforcement learning
STATE_SIZE = 10  # Define number of states
ACTION_SIZE = 4  # Define number of actions
Q_TABLE = np.zeros((STATE_SIZE, ACTION_SIZE))  # Initialize Q-table
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1  # For exploration

# Function for handling Reinforcement Learning (RL)
async def handle_reinforcement_learning(message):
    """
    Handles RL by updating the Q-table based on user interaction and bot responses.
    """
    state = random.randint(0, STATE_SIZE - 1)  # Simulate state (you can refine based on actual user input)
    action = np.argmax(Q_TABLE[state]) if random.uniform(0, 1) > EPSILON else random.randint(0, ACTION_SIZE - 1)

    # Simulate reward based on the action taken
    if action == 0:  # Action: respond helpfully
        reward = 1  # Positive reward
        await message.channel.send("I'm here to help you!")
    elif action == 1:  # Action: provide sentiment response
        reward = 0.5  # Neutral reward
        await message.channel.send("Thanks for interacting!")
    elif action == 2:  # Action: provide a random fun fact
        reward = 0.2  # Neutral reward
        await message.channel.send("Did you know? Elephants can't jump!")
    else:  # Action: joke
        reward = -0.1  # Negative reward
        await message.channel.send("Why don’t skeletons fight each other? They don’t have the guts!")

    # Q-learning update
    new_state = random.randint(0, STATE_SIZE - 1)  # Simulate the next state
    Q_TABLE[state, action] = (1 - LEARNING_RATE) * Q_TABLE[state, action] + \
                             LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_TABLE[new_state]))

    print(f"Q-Table Updated: {Q_TABLE}")

# Bot Events
@bot.event
async def on_ready():
    print(f'{bot.user.name} is online and ready to assist!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Sentiment analysis on mentions
    if bot.user in message.mentions:
        sentiment = sentiment_analyzer(message.content)[0]
        await message.channel.send('You mentioned me!')
        await message.channel.send(f'Sentiment detected: {sentiment["label"]} with a score of {sentiment["score"]:.2f}')
        
        # Trigger Reinforcement Learning state-action update
        await handle_reinforcement_learning(message)
    
    await bot.process_commands(message)

# Commands
@bot.command()
async def play(ctx, *, song_name):
    """
    Plays a song by searching for it on Spotify.
    """
    try:
        results = sp.search(q=song_name, limit=1)
        if results['tracks']['items']:
            song = results['tracks']['items'][0]['name']
            artist = results['tracks']['items'][0]['artists'][0]['name']
            await ctx.send(f'Now playing: {song} by {artist}')
        else:
            await ctx.send('Song not found on Spotify.')
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def recommend_streamers(ctx):
    """
    Recommends a list of streamers.
    """
    streamers = ['Streamer1', 'Streamer2', 'Streamer3']
    await ctx.send(f'Check out these Twitch streamers: {", ".join(streamers)}')

@bot.command()
async def news(ctx):
    """
    Fetches the latest technology news from an API.
    """
    try:
        news_api_key = 'your_news_api_key_here'  # Replace with your actual News API key
        news_url = f'https://newsapi.org/v2/top-headlines?category=technology&apiKey={news_api_key}'
        response = requests.get(news_url)
        articles = response.json()['articles']
        headlines = [article['title'] for article in articles]
        await ctx.send('\n'.join(headlines[:5]))
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def custom_help(ctx):
    """
    Displays a help message.
    """
    await ctx.send("Here are some commands you can try:\n"
                "/play [song name] - Plays a song on Spotify\n"
                "/recommend_streamers - Recommends streamers\n"
                "/news - Shows latest tech news\n"
                "Just mention me for a chat!")

# Run the bot
bot.run(DISCORD_TOKEN)
