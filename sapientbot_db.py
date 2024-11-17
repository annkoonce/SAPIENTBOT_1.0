# sapientbot_db.py

import sqlite3
from datetime import datetime, timedelta

# ============================
# DATABASE CONNECTION
# ============================
def connect_db():
    """Establish and return a database connection and cursor."""
    conn = sqlite3.connect('sapientbot.db')
    cursor = conn.cursor()
    return conn, cursor

# ============================
# USER MANAGEMENT FUNCTIONS
# ============================
def add_or_update_user(user_id):
    """Add a user or update existing user details."""
    conn, cursor = connect_db()
    cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()

def get_user_data(user_id):
    """Retrieve user data."""
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    data = cursor.fetchone()
    conn.close()
    return data

def update_points(user_id, points):
    """Update user points."""
    conn, cursor = connect_db()
    cursor.execute(
        "UPDATE users SET points = points + ? WHERE user_id = ?",
        (points, user_id)
    )
    conn.commit()
    conn.close()

def get_user_badges(user_id):
    """Retrieve user badges."""
    conn, cursor = connect_db()
    cursor.execute("SELECT badges FROM users WHERE user_id = ?", (user_id,))
    badges = cursor.fetchone()
    conn.close()
    return badges[0] if badges else ""

def assign_badge(user_id, badge):
    """Assign a badge to a user."""
    badges = get_user_badges(user_id)
    if badge not in badges:
        badges = f"{badges},{badge}" if badges else badge
        conn, cursor = connect_db()
        cursor.execute(
            "UPDATE users SET badges = ? WHERE user_id = ?",
            (badges, user_id)
        )
        conn.commit()
        conn.close()

def award_daily_bonus(user_id):
    """Award a daily bonus to the user."""
    conn, cursor = connect_db()
    cursor.execute("SELECT last_reward_time FROM users WHERE user_id = ?", (user_id,))
    last_reward = cursor.fetchone()
    now = datetime.now()

    if last_reward and (now - datetime.strptime(last_reward[0], '%Y-%m-%d %H:%M:%S')).days < 1:
        return False  # Bonus already awarded today

    cursor.execute(
        "UPDATE users SET points = points + 50, last_reward_time = ? WHERE user_id = ?",
        (now.strftime('%Y-%m-%d %H:%M:%S'), user_id)
    )
    conn.commit()
    conn.close()
    return True

def get_leaderboard(limit=10):
    """Get the top users for a leaderboard."""
    conn, cursor = connect_db()
    cursor.execute("SELECT user_id, points FROM users ORDER BY points DESC LIMIT ?", (limit,))
    leaderboard = cursor.fetchall()
    conn.close()
    return leaderboard

def analyze_feedback(threshold=3):
    """
    Retrieve all feedback with a rating below a given threshold.
    """
    conn, cursor = connect_db()
    cursor.execute(
        "SELECT query, response, feedback FROM feedback WHERE feedback < ?",
        (threshold,)
    )
    bad_responses = cursor.fetchall()
    conn.close()
    return bad_responses

# ============================
# FEEDBACK & PREFERENCES
# ============================
def create_feedback_table():
    """Create the feedback table."""
    conn, cursor = connect_db()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            response TEXT,
            feedback INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_preferences_table():
    """Create the user preferences table."""
    conn, cursor = connect_db()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preference_key TEXT,
            preference_value TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_feedback(user_id, query, response, feedback):
    """Save user feedback into the database."""
    conn, cursor = connect_db()
    cursor.execute(
        "INSERT INTO feedback (user_id, query, response, feedback) VALUES (?, ?, ?, ?)",
        (user_id, query, response, feedback)
    )
    conn.commit()
    conn.close()

def set_user_preference(user_id, key, value):
    """Set a user preference."""
    conn, cursor = connect_db()
    cursor.execute(
        "INSERT OR REPLACE INTO user_preferences (user_id, preference_key, preference_value) VALUES (?, ?, ?)",
        (user_id, key, value)
    )
    conn.commit()
    conn.close()

# ============================
# INITIALIZATION
# ============================
def initialize_advanced_features():
    """Initialize advanced database features."""
    create_feedback_table()
    create_preferences_table()

# Ensure the database is ready
initialize_advanced_features()
