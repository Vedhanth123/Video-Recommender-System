"""
Synthetic Data Generator for Emotion-Based Video Recommendation system.
This script creates synthetic data for training the neural network model.

It generates data for the following tables:
- emotion_logs: User emotion detection logs
- video_clicks: Video click interactions
- video_interactions: Video category/emotion interactions
"""

import sqlite3
import random
import time
from datetime import datetime, timedelta
import os

# Configuration
DATABASE_PATH = "user_data.db"
NUM_USERS = 5  # Number of synthetic users
NUM_EMOTION_LOGS = 50  # Number of emotion logs to generate
NUM_VIDEO_CLICKS = 20  # Number of video clicks to generate
NUM_VIDEO_INTERACTIONS = 30  # Number of video interactions to generate

# YouTube video categories mapping
YOUTUBE_CATEGORIES = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles", 
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers"
}

# Emotions that our system can detect
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]

# Emotion to video mapping (borrowed from emotion_constants.py)
EMOTION_VIDEO_MAPPING = {
    "happy": {
        "keywords": ["uplifting music", "comedy videos", "funny moments", "feel good content"],
        "categories": ["22", "23", "24"],  # Music, Comedy, Entertainment
    },
    "sad": {
        "keywords": ["relaxing music", "calming videos", "motivational speeches", "inspirational stories"],
        "categories": ["22", "25", "27"],  # Music, News & Politics, Education
    },
    "angry": {
        "keywords": ["calming music", "meditation videos", "relaxation techniques", "nature sounds"],
        "categories": ["22", "28"],  # Music, Science & Technology
    },
    "fear": {
        "keywords": ["soothing music", "positive affirmations", "calming content", "guided relaxation"],
        "categories": ["22", "26"],  # Music, Howto & Style
    },
    "surprise": {
        "keywords": ["amazing facts", "incredible discoveries", "wow moments", "mind blowing"],
        "categories": ["28", "24", "27"],  # Science & Tech, Entertainment, Education
    },
    "neutral": {
        "keywords": ["interesting documentaries", "educational content", "informative videos", "how-to guides"],
        "categories": ["27", "28", "26"],  # Education, Science & Tech, Howto & Style
    },
    "disgust": {
        "keywords": ["satisfying videos", "clean organization", "aesthetic content", "art videos"],
        "categories": ["24", "26"],  # Entertainment, Howto & Style
    }
}

# Sample video titles for each category
VIDEO_TITLES = {
    "22": ["Daily Vlog: My Weekend Adventures", "Meet My Family", "Life Updates and News", "My Morning Routine"],
    "23": ["Stand-up Comedy Special", "Funny Pranks Compilation", "Comedy Sketches That Made Me Laugh", "Humorous Moments"],
    "24": ["Top 10 Celebrity Moments", "Amazing Entertainment News", "Talent Show Highlights", "Behind the Scenes"],
    "25": ["Breaking News Today", "Political Analysis", "Current Events Discussion", "Global News Updates"],
    "26": ["How to Cook Perfect Pasta", "5 Fashion Tips for Summer", "DIY Home Decor Ideas", "Beauty Tutorial"],
    "27": ["Learning Math Made Easy", "History Documentary", "Science for Beginners", "Educational Series"],
    "28": ["Latest Tech News", "Science Experiments at Home", "How Technology Works", "Future Tech Predictions"],
    "10": ["Top Hits Music Mix", "Relaxing Piano Music", "Upbeat Workout Playlist", "Acoustic Cover Songs"],
    "20": ["Minecraft Let's Play", "Gaming Highlights", "Game Review", "Speedrun Challenge"]
}

# Sample video IDs (these are made up for synthetic data)
VIDEO_IDS = [
    "dQw4w9WgXcQ", "J_8mdH20qTQ", "9bZkp7q19f0", "kJQP7kiw5Fk", 
    "OPf0YbXqDm0", "fJ9rUzIMcZQ", "6Ejga4kJUts", "lWA2pjMjpBs",
    "YR5ApYxkU-U", "CdXesX6mYUE", "weeI1G46q0o", "g3AdacM7Sf8",
    "y6120QOlsfU", "PWgvGjAhvIw", "l_jWlkh-sHU", "CPy2TLPbpvY"
]

# Sample locations
LOCATIONS = [
    ("New York", "USA"),
    ("London", "UK"),
    ("Tokyo", "Japan"),
    ("Paris", "France"),
    ("Sydney", "Australia"),
    ("Toronto", "Canada"),
    ("Berlin", "Germany"),
    ("Mumbai", "India"),
    ("Unknown City", "Unknown Country")
]

# Generate synthetic user names
def generate_users(n):
    """Generate n synthetic user names"""
    first_names = ["Alex", "Jordan", "Casey", "Taylor", "Morgan", "Riley", "Jamie", "Avery", "Skyler", "Dakota"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
    
    users = []
    for _ in range(n):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        users.append(name)
    
    return users

# Ensure all necessary database tables exist
def ensure_database_tables():
    """Create necessary database tables if they don't exist"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create emotion_logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        emotion TEXT,
        input TEXT,
        timestamp INTEGER,
        city TEXT,
        country TEXT
    )
    ''')
    
    # Create video_clicks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS video_clicks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        video_id TEXT,
        video_title TEXT,
        video_category TEXT,
        emotion TEXT,
        timestamp INTEGER,
        city TEXT,
        country TEXT
    )
    ''')
    
    # Create video_interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS video_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        video_genre TEXT,
        emotion TEXT,
        timestamp INTEGER,
        city TEXT,
        country TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database tables created/verified.")

# Generate synthetic emotion logs
def generate_emotion_logs(users, num_logs):
    """Generate synthetic emotion logs for the given users"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print(f"Generating {num_logs} emotion logs...")
    
    # Generate random logs within the last 30 days
    now = datetime.now()
    records = []
    
    for _ in range(num_logs):
        user = random.choice(users)
        emotion = random.choice(EMOTIONS)
        
        # Random time in the last 30 days
        days_back = random.randint(0, 30)
        hours_back = random.randint(0, 23)
        random_time = now - timedelta(days=days_back, hours=hours_back)
        timestamp = int(random_time.timestamp())
        
        # Random location
        city, country = random.choice(LOCATIONS)
        
        # Sample input text based on emotion
        input_text = f"I feel {emotion} today"
        
        # Insert into database
        cursor.execute('''
        INSERT INTO emotion_logs (name, emotion, input, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user, emotion, input_text, timestamp, city, country))
        
        records.append((user, emotion, timestamp))
    
    conn.commit()
    conn.close()
    print(f"Generated {num_logs} emotion logs.")
    return records

# Generate synthetic video clicks
def generate_video_clicks(users, emotion_logs, num_clicks):
    """Generate synthetic video clicks for the given users"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print(f"Generating {num_clicks} video clicks...")
    
    for _ in range(num_clicks):
        # Get a random user and their emotion log
        log_index = random.randint(0, len(emotion_logs) - 1)
        user, emotion, timestamp = emotion_logs[log_index]
        
        # Video click happens after emotion log (0-60 minutes later)
        click_timestamp = timestamp + random.randint(60, 3600)
        
        # Get video category appropriate for the emotion
        if random.random() < 0.8:  # 80% chance to pick matching category
            # Get a category that matches the emotion
            if emotion in EMOTION_VIDEO_MAPPING:
                category_id = random.choice(EMOTION_VIDEO_MAPPING[emotion]["categories"])
            else:
                category_id = random.choice(list(YOUTUBE_CATEGORIES.keys()))
        else:
            # Just pick a random category
            category_id = random.choice(list(YOUTUBE_CATEGORIES.keys()))
            
        # Get category name
        category_name = YOUTUBE_CATEGORIES.get(category_id, "Other")
        
        # Get a random video title appropriate for the category
        if category_id in VIDEO_TITLES:
            title = random.choice(VIDEO_TITLES[category_id])
        else:
            # Generic title with category name
            title = f"Video about {category_name}"
            
        # Get a random video ID
        video_id = random.choice(VIDEO_IDS)
        
        # Random location (usually the same as emotion log)
        if random.random() < 0.9:  # 90% chance to have same location
            cursor.execute('''
            SELECT city, country FROM emotion_logs WHERE name=? AND timestamp=?
            ''', (user, timestamp))
            result = cursor.fetchone()
            if result:
                city, country = result
            else:
                city, country = random.choice(LOCATIONS)
        else:
            city, country = random.choice(LOCATIONS)
        
        # Insert into database
        cursor.execute('''
        INSERT INTO video_clicks (name, video_id, video_title, video_category, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user, video_id, title, category_id, emotion, click_timestamp, city, country))
    
    conn.commit()
    conn.close()
    print(f"Generated {num_clicks} video clicks.")

# Generate synthetic video interactions
def generate_video_interactions(users, emotion_logs, num_interactions):
    """Generate synthetic video interactions for the given users"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print(f"Generating {num_interactions} video interactions...")
    
    for _ in range(num_interactions):
        # Get a random user and their emotion log
        log_index = random.randint(0, len(emotion_logs) - 1)
        user, emotion, timestamp = emotion_logs[log_index]
        
        # Interaction happens after emotion log (0-60 minutes later)
        interaction_timestamp = timestamp + random.randint(60, 3600)
        
        # Get video genre appropriate for the emotion
        if random.random() < 0.8:  # 80% chance to pick matching category
            # Get a category that matches the emotion
            if emotion in EMOTION_VIDEO_MAPPING:
                category_id = random.choice(EMOTION_VIDEO_MAPPING[emotion]["categories"])
            else:
                category_id = random.choice(list(YOUTUBE_CATEGORIES.keys()))
        else:
            # Just pick a random category
            category_id = random.choice(list(YOUTUBE_CATEGORIES.keys()))
            
        # Get category name (this is the video genre)
        video_genre = YOUTUBE_CATEGORIES.get(category_id, "Other")
        
        # Random location (usually the same as emotion log)
        if random.random() < 0.9:  # 90% chance to have same location
            cursor.execute('''
            SELECT city, country FROM emotion_logs WHERE name=? AND timestamp=?
            ''', (user, timestamp))
            result = cursor.fetchone()
            if result:
                city, country = result
            else:
                city, country = random.choice(LOCATIONS)
        else:
            city, country = random.choice(LOCATIONS)
        
        # Insert into database
        cursor.execute('''
        INSERT INTO video_interactions (name, video_genre, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user, video_genre, emotion, interaction_timestamp, city, country))
    
    conn.commit()
    conn.close()
    print(f"Generated {num_interactions} video interactions.")

def main():
    """Main function to generate all synthetic data"""
    print("Starting synthetic data generation...")
    
    # Ensure database tables exist
    ensure_database_tables()
    
    # Generate synthetic users
    users = generate_users(NUM_USERS)
    print(f"Generated {len(users)} synthetic users: {users}")
    
    # Generate emotion logs
    emotion_logs = generate_emotion_logs(users, NUM_EMOTION_LOGS)
    
    # Generate video clicks
    generate_video_clicks(users, emotion_logs, NUM_VIDEO_CLICKS)
    
    # Generate video interactions
    generate_video_interactions(users, emotion_logs, NUM_VIDEO_INTERACTIONS)
    
    print("Synthetic data generation complete!")
    print(f"Database file: {os.path.abspath(DATABASE_PATH)}")

if __name__ == "__main__":
    main()
