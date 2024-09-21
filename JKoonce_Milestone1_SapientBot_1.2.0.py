import os
import discord
from discord.ext import commands
import random
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from flask import Flask, request
import threading
import webbrowser
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Bot Token and API keys
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

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

# Spotify Authentication for public commands (searching, top tracks)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Reinforcement Learning Module
class RLModule:
    def __init__(self):
        self.q_table = {}  # Holds state-action values
        self.actions = ['reward_user', 'ignore', 'provide_tip']
    
    def update_q_table(self, state, action, reward):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        self.q_table[state][action] += reward
    
    def choose_action(self, state):
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return random.choice(self.actions)

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
rl_module = RLModule()
gamification = GamificationSystem()

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
        await ctx.send("You need to authorize Spotify first using `!authorize_spotify`.")
        return
    sp_user = spotipy.Spotify(auth=spotify_token)
    devices = sp_user.devices()
    if not devices['devices']:
        await ctx.send("No active Spotify devices found.")
    else:
        sp_user.start_playback()
        await ctx.send("Playback started!")

# 6. !spotify_pause: Pauses playback on an active Spotify device.
@bot.command(name='spotify_pause')
async def spotify_pause(ctx):
    """Pauses playback on an active Spotify device."""
    if spotify_token is None:
        await ctx.send("You need to authorize Spotify first using `!authorize_spotify`.")
        return
    sp_user = spotipy.Spotify(auth=spotify_token)
    sp_user.pause_playback()
    await ctx.send("Playback paused.")

# 7. !spotify_next: Skips to the next track on an active Spotify device.
@bot.command(name='spotify_next')
async def spotify_next(ctx):
    """Skips to the next track on an active Spotify device."""
    if spotify_token is None:
        await ctx.send("You need to authorize Spotify first using `!authorize_spotify`.")
        return
    sp_user = spotipy.Spotify(auth=spotify_token)
    sp_user.next_track()
    await ctx.send("Playing next track.")

# 8. !spotify_previous: Goes to the previous track on an active Spotify device.
@bot.command(name='spotify_previous')
async def spotify_previous(ctx):
    """Goes to the previous track on an active Spotify device."""
    if spotify_token is None:
        await ctx.send("You need to authorize Spotify first using `!authorize_spotify`.")
        return
    sp_user = spotipy.Spotify(auth=spotify_token)
    sp_user.previous_track()
    await ctx.send("Playing previous track.")

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
    elif action == 'provide_tip':
        await ctx.send(f"Tip: Keep engaging with the bot to earn more points and rewards!")
    else:
        await ctx.send(f"Here's your profile, {ctx.author.name}:\nTotal points: {total_points}, Level: {level}, Badges: {badge_str}")

# 2. !reward: Rewards the user with points and checks quest progress.
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
    for rank, (username, points) in enumerate(leaderboard, 1):
        message += f"{rank}. {username} - {points} points\n"

        # Apply RL action to possibly reward the leaderboard users
        user_state = f"leaderboard_{username}"
        action = rl_module.choose_action(user_state)
        if action == 'reward_user':
            gamification.reward_user(username, 5)  # Assuming usernames are keys in user_rewards
            message += f"{username} received a 5-point reward!\n"
        elif action == 'provide_tip':
            message += f"{username}, keep up the good work to stay on top!\n"

    await ctx.send(message)


# Error Handling
@bot.event
async def on_command_error(ctx, error):
    """Handles errors and provides feedback to the user."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Command not found. Type `!commands` to see the list of available commands.")
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

# Run the bot
bot.run(DISCORD_TOKEN)
