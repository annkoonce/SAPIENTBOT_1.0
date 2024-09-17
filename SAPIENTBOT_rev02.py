import os
import discord
import random
import asyncio
import numpy as np
import requests
from collections import defaultdict
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
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Q-table for reinforcement learning
STATE_SIZE = 10  # Define number of states
ACTION_SIZE = 4  # Define number of actions
Q_TABLE = np.zeros((STATE_SIZE, ACTION_SIZE))  # Initialize Q-table
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1  # For exploration

# Tracking user messages for spam detection
user_message_count = defaultdict(int)
user_points = defaultdict(int)

# Function for handling Reinforcement Learning (RL)
async def handle_reinforcement_learning(message):
    state = random.randint(0, STATE_SIZE - 1)  # Simulate state (you can refine based on actual user input)
    action = np.argmax(Q_TABLE[state]) if random.uniform(0, 1) > EPSILON else random.randint(0, ACTION_SIZE - 1)

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

    user_message_count[message.author.id] += 1
    if user_message_count[message.author.id] > 5:
        await message.channel.send(f"{message.author.mention}, please stop spamming.")
        await message.author.add_roles(discord.utils.get(message.guild.roles, name="Muted"))

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
    try:
        results = sp.search(q=song_name, limit=1)
        if results['tracks']['items']:
            song = results['tracks']['items'][0]['name']
            artist = results['tracks']['items'][0]['artists'][0]['name']
            song_url = results['tracks']['items'][0]['external_urls']['spotify']
            await ctx.send(f'Now playing: {song} by {artist}\n{song_url}')
        else:
            await ctx.send('Song not found on Spotify.')
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def recommend_streamers(ctx):
    streamers = ['KilanaAnn', 'theHighHeeledGamer', 'AthenaAllianceCLT','ImRunninBull','DreamWarrior']
    selected_streamer = random.choice(streamers)
    await ctx.send(f'Check out this Twitch streamer: https://twitch.tv/{selected_streamer}')

@bot.command()
async def news(ctx):
    try:
        news_api_key = os.getenv('NEWS_API_KEY')
        news_url = f'https://newsapi.org/v2/top-headlines?category=technology&apiKey={news_api_key}'
        response = requests.get(news_url)
        articles = response.json()['articles']
        headlines = [article['title'] for article in articles]
        await ctx.send('\n'.join(headlines[:5]))
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def custom_help(ctx):
    await ctx.send("Here are some commands you can try:\n"
                "/play [song name] - Plays a song on Spotify\n"
                "/recommend_streamers - Recommends streamers\n"
                "/news - Shows latest tech news\n"
                "Just mention me for a chat!")

# Reaction Roles
@bot.command()
async def setup_roles(ctx):
    message = await ctx.send("React to this message to assign yourself a role.")
    reactions = ['🎮', '🎧', '🛠️']
    for reaction in reactions:
        await message.add_reaction(reaction)

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    if str(reaction.emoji) == '🎮':
        role = discord.utils.get(user.guild.roles, name="Gamer")
        await user.add_roles(role)
    elif str(reaction.emoji) == '🎧':
        role = discord.utils.get(user.guild.roles, name="Music Lover")
        await user.add_roles(role)

# Moderation Commands
@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    await member.ban(reason=reason)
    await ctx.send(f'{member.mention} has been banned for {reason}.')

@bot.command()
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    await member.kick(reason=reason)
    await ctx.send(f'{member.mention} has been kicked for {reason}.')

# Poll System
@bot.command()
async def poll(ctx, question, *options):
    if len(options) > 10:
        await ctx.send("You can only provide up to 10 options.")
        return
    if len(options) < 2:
        await ctx.send("You need to provide at least 2 options.")
        return

    poll_message = await ctx.send(f"**{question}**\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]))
    for i in range(len(options)):
        await poll_message.add_reaction(f'{i+1}\u20E3')

# Welcome Message
@bot.event
async def on_member_join(member):
    welcome_channel = bot.get_channel('YOUR_WELCOME_CHANNEL_ID')
    await welcome_channel.send(f"Welcome {member.mention} to the server! Feel free to introduce yourself.")
    add_points(str(member.id), 10)
    await welcome_channel.send(f"{member.mention} earned 10 points for joining! Your total is now {user_points.get(member.id, 10)}.")

# Reminder System
@bot.command()
async def set_reminder(ctx, time: int, *, reminder: str):
    await ctx.send(f"Reminder set! I'll remind you in {time} minutes.")
    await asyncio.sleep(time * 60)
    await ctx.send(f"Reminder: {reminder}")

# Run the bot
bot.run(DISCORD_TOKEN)
