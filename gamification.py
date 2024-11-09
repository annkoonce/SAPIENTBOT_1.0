# gamification.py

import sqlite3
from datetime import datetime, timedelta

class GamificationSystem:
    def __init__(self, db_file='sapientbot_gamification.db'):
        self.db_file = db_file
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()
        self.create_tables()
        self.badge_criteria = {
            "Novice": 100,
            "Intermediate": 300,
            "Advanced": 500,
            "Expert": 1000,
            "Master": 2000
        }

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            points INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            badges TEXT,
            last_reward_time TEXT
        )''')
        self.connection.commit()

    def reward_user(self, user_id, points):
        user_data = self.get_user_data(user_id)
        if not user_data:
            self.cursor.execute('''INSERT INTO users (user_id, points, level, badges, last_reward_time)
                                VALUES (?, ?, ?, ?, ?)''', 
                                (user_id, points, 1, '', datetime.now().isoformat()))
        else:
            new_points = user_data['points'] + points
            self.cursor.execute('''UPDATE users SET points = ?, last_reward_time = ? 
                                WHERE user_id = ?''', 
                                (new_points, datetime.now().isoformat(), user_id))
            self.check_level_up(user_id, new_points)
            self.assign_badges(user_id, new_points)
        self.connection.commit()

    def get_user_data(self, user_id):
        self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                'user_id': row[0],
                'points': row[1],
                'level': row[2],
                'badges': row[3].split(',') if row[3] else [],
                'last_reward_time': row[4]
            }
        return None

    def check_level_up(self, user_id, points):
        new_level = points // 100
        self.cursor.execute('UPDATE users SET level = ? WHERE user_id = ?', (new_level, user_id))
        self.connection.commit()

    def assign_badges(self, user_id, points):
        user_data = self.get_user_data(user_id)
        existing_badges = set(user_data['badges'])
        new_badges = set()

        for badge, threshold in self.badge_criteria.items():
            if points >= threshold and badge not in existing_badges:
                new_badges.add(badge)

        if new_badges:
            updated_badges = existing_badges.union(new_badges)
            self.cursor.execute('UPDATE users SET badges = ? WHERE user_id = ?',
                                (','.join(updated_badges), user_id))
            self.connection.commit()

    def get_badges(self, user_id):
        user_data = self.get_user_data(user_id)
        return user_data['badges'] if user_data else []

    def close_connection(self):
        self.connection.close()
