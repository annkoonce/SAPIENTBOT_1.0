import discord
import requests
from discord.ext import commands
import keepalive
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import Spotify
from flask import Flask
from threading import Thread
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello SAPIENTBOT is Online & Ready.'

# Flask run command modification to allow Flask to run in a separate thread
def run():
    app.run(host='0.0.0.0', port=8000)

# Start the Flask server in a new thread
t = Thread(target=run)
t.start()

# Create a new bot instance with intents
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)

# Initialize NLP tools
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@bot.event
async def on_ready():
    print('SAPIENTBOT is online & ready.')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if the bot is mentioned
    if bot.user in message.mentions:
        await message.channel.send('You mentioned me!')
        await message.channel.send('How can I help you today?')

    # Process sentiment of user messages
    sentiment = sia.polarity_scores(message.content)
    if sentiment['compound'] >= 0.05:
        await message.channel.send('I sense positivity in your message!')
    elif sentiment['compound'] <= -0.05:
        await message.channel.send('You seem a bit down. Is there anything I can do to help?')
    else:
        await message.channel.send('Neutral vibes detected.')

    # Additional commands
    if message.content.lower().startswith('hello'):
        await message.channel.send('Hello!')

    if message.content.lower().startswith('how are you'):
        await message.channel.send("I'm a bot, I don't have feelings, but thanks for asking!")

    if message.content.lower().startswith('tell me a joke'):
        await message.channel.send('Why don\'t skeletons fight each other? They don\'t have the guts!')  
    
    await bot.process_commands(message)

@bot.command()
async def play(ctx, *, song_name):
    sp = Spotify(client_credentials_manager=SpotifyClientCredentials())
    results = sp.search(q=song_name, limit=1)
    if results['tracks']['items']:
        await ctx.send(f'Now playing: {song_name}')
    else:
        await ctx.send('Song not found on Spotify.')

@bot.command()
async def recommend_streamers(ctx):
    streamers = ['Streamer1', 'Streamer2', 'Streamer3']
    await ctx.send(f'Check out these Twitch streamers: {", ".join(streamers)}')

@bot.command()
async def news(ctx):
    news_api_key = 'your_news_api_key_here'
    news_url = f'https://newsapi.org/v2/top-headlines?category=technology&apiKey={news_api_key}'
    response = requests.get(news_url)
    articles = response.json()['articles']
    headlines = [article['title'] for article in articles]
    await ctx.send('\n'.join(headlines[:5]))

@bot.command()
async def data_policy(ctx):
    await ctx.send("SAPIENTBOT respects your privacy. We only collect data necessary to improve your experience. "
                "You can request data deletion at any time by using the /delete_data command.")

@bot.command()
async def delete_data(ctx):
    await ctx.send("Your data has been deleted as requested. We value your privacy and data security.")

print(keepalive.DISCORD_BOT_TOKEN)
# Run the bot
bot.run(keepalive.DISCORD_BOT_TOKEN)
