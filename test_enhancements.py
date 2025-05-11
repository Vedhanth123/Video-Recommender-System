"""
Test module for the recommendation enhancement system.
This script tests the enhanced recommendation features.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime
import random

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from recommendation_enhancement import RecommendationEnhancer
from config import DATABASE_PATH

def create_test_videos():
    """Create a set of test videos with various properties."""
    videos = []
    
    # Morning-appropriate videos
    videos.append({
        'id': 'morning1',
        'title': 'Morning Energizing Workout Routine',
        'description': 'Start your day with this energizing workout routine to boost your energy.',
        'views': 10000,
        'channel': 'FitnessChannel',
        'duration': '15:30',
        'thumbnail': 'https://example.com/thumbnail1.jpg',
        'category': 'Fitness'
    })
    
    videos.append({
        'id': 'morning2',
        'title': 'Motivational Speech for a Fresh Start',
        'description': 'Listen to this motivational speech to kickstart your day with positivity.',
        'views': 8500,
        'channel': 'MotivationDaily',
        'duration': '10:15',
        'thumbnail': 'https://example.com/thumbnail2.jpg',
        'category': 'Motivation'
    })
    
    # Afternoon-appropriate videos
    videos.append({
        'id': 'afternoon1',
        'title': 'Educational Documentary: History of Space Exploration',
        'description': 'Learn about the fascinating history of space exploration in this informative documentary.',
        'views': 15000,
        'channel': 'EduDocs',
        'duration': '45:20',
        'thumbnail': 'https://example.com/thumbnail3.jpg',
        'category': 'Education'
    })
    
    videos.append({
        'id': 'afternoon2',
        'title': 'Relaxing Piano Music for Work and Study',
        'description': 'Enhance your productivity with this relaxing piano music perfect for afternoon work sessions.',
        'views': 12000,
        'channel': 'RelaxMusicHub',
        'duration': '2:30:00',
        'thumbnail': 'https://example.com/thumbnail4.jpg',
        'category': 'Music'
    })
    
    # Evening-appropriate videos
    videos.append({
        'id': 'evening1',
        'title': 'Stand-up Comedy Special: Laugh Till You Drop',
        'description': 'Wind down your day with this hilarious comedy special that will have you in stitches.',
        'views': 25000,
        'channel': 'ComedyCentral',
        'duration': '1:05:45',
        'thumbnail': 'https://example.com/thumbnail5.jpg',
        'category': 'Comedy'
    })
    
    videos.append({
        'id': 'evening2',
        'title': 'Evening Yoga for Relaxation',
        'description': 'Unwind after a long day with this gentle yoga session designed to help you relax.',
        'views': 18000,
        'channel': 'YogaWithMe',
        'duration': '35:10',
        'thumbnail': 'https://example.com/thumbnail6.jpg',
        'category': 'Yoga'
    })
    
    # Night-appropriate videos
    videos.append({
        'id': 'night1',
        'title': 'Calming Sleep Music with Ocean Sounds',
        'description': 'Fall asleep faster with this peaceful ocean soundscape designed to promote deep sleep.',
        'views': 30000,
        'channel': 'SleepSounds',
        'duration': '8:00:00',
        'thumbnail': 'https://example.com/thumbnail7.jpg',
        'category': 'Sleep'
    })
    
    videos.append({
        'id': 'night2',
        'title': 'Meditation for Peaceful Sleep',
        'description': 'This guided meditation will help quiet your mind and prepare you for restful sleep.',
        'views': 22000,
        'channel': 'MeditationMaster',
        'duration': '25:30',
        'thumbnail': 'https://example.com/thumbnail8.jpg',
        'category': 'Meditation'
    })
    
    # Add some neutral videos
    videos.append({
        'id': 'neutral1',
        'title': 'Top 10 Travel Destinations for 2025',
        'description': 'Discover the most beautiful and trending travel destinations for your next vacation.',
        'views': 35000,
        'channel': 'TravelGuide',
        'duration': '18:45',
        'thumbnail': 'https://example.com/thumbnail9.jpg',
        'category': 'Travel'
    })
    
    videos.append({
        'id': 'neutral2',
        'title': 'Ultimate Guide to Photography',
        'description': 'Learn professional photography techniques with this comprehensive guide.',
        'views': 28000,
        'channel': 'PhotoPro',
        'duration': '55:20',
        'thumbnail': 'https://example.com/thumbnail10.jpg',
        'category': 'Photography'
    })
    
    return videos

def test_time_of_day_scoring():
    """Test the time-of-day awareness feature."""
    print("\n=== Testing Time-of-Day Scoring ===")
    
    # Create enhancer
    enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    # Create test videos
    videos = create_test_videos()
    
    # Test different times
    test_hours = {
        "morning": 8,     # 8 AM
        "afternoon": 15,  # 3 PM
        "evening": 19,    # 7 PM
        "night": 23       # 11 PM
    }
    
    for time_segment, hour in test_hours.items():
        print(f"\nTesting {time_segment.upper()} ({hour}:00):")
        print("-" * 40)
        
        # Score each video for this time
        for video in videos:
            score = enhancer.time_of_day_score(video, hour)
            print(f"{video['title'][:30]}... Score: {score:.2f}")
        
        # Find best videos for this time
        scored_videos = [(video, enhancer.time_of_day_score(video, hour)) for video in videos]
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 3 videos for this time:")
        for video, score in scored_videos[:3]:
            print(f"- {video['title']} (Score: {score:.2f})")

def test_collaborative_filtering():
    """Test the collaborative filtering feature."""
    print("\n=== Testing Collaborative Filtering ===")
    
    # Create enhancer
    enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    # Create test videos
    videos = create_test_videos()
    
    # Get existing users from database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT name FROM video_clicks LIMIT 5")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if users:
            print(f"Found {len(users)} users for testing: {', '.join(users)}")
            
            # Test collaborative filtering for first user
            test_user = users[0]
            print(f"\nTesting collaborative filtering for user: {test_user}")
            
            # Before filtering
            random.shuffle(videos)
            print("\nBefore collaborative filtering:")
            for i, video in enumerate(videos[:5]):
                print(f"{i+1}. {video['title']}")
            
            # Apply collaborative filtering
            filtered_videos = enhancer.collaborative_filtering(test_user, videos.copy())
            
            # After filtering
            print("\nAfter collaborative filtering:")
            for i, video in enumerate(filtered_videos[:5]):
                collab_score = video.get('collab_score', 0)
                print(f"{i+1}. {video['title']} (Collab Score: {collab_score:.2f})")
        else:
            print("No users found in the database for testing collaborative filtering.")
    except Exception as e:
        print(f"Error testing collaborative filtering: {e}")

def test_reranking():
    """Test the overall reranking algorithm."""
    print("\n=== Testing Reranking Algorithm ===")
    
    # Create enhancer
    enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    # Create test videos
    videos = create_test_videos()
    
    # Get existing users from database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT name FROM video_clicks LIMIT 5")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if users:
            # Test reranking for first user
            test_user = users[0]
            current_hour = datetime.now().hour
            
            print(f"\nTesting reranking for user: {test_user}, Hour: {current_hour}")
            
            # Before reranking
            random.shuffle(videos)
            print("\nBefore reranking:")
            for i, video in enumerate(videos[:5]):
                print(f"{i+1}. {video['title']}")
            
            # Apply reranking
            reranked_videos = enhancer.rerank_recommendations(test_user, videos.copy(), current_hour=current_hour)
            
            # After reranking
            print("\nAfter reranking:")
            for i, video in enumerate(reranked_videos[:5]):
                final_score = video.get('final_score', 0)
                print(f"{i+1}. {video['title']} (Final Score: {final_score:.2f})")
                
                # Show individual score components
                time_score = video.get('time_score', 0)
                collab_score = min(video.get('collab_score', 0) / 5.0, 1.0)
                print(f"   - Time Score: {time_score:.2f}, Collab Score: {collab_score:.2f}")
        else:
            print("No users found in the database for testing reranking.")
    except Exception as e:
        print(f"Error testing reranking: {e}")

def test_content_diversity():
    """Test the content diversity feature."""
    print("\n=== Testing Content Diversity ===")
    
    # Create enhancer
    enhancer = RecommendationEnhancer(DATABASE_PATH)
    
    # Create videos with limited diversity
    videos = []
    
    # Add 5 videos with the same category
    for i in range(5):
        videos.append({
            'id': f'fitness{i}',
            'title': f'Fitness Workout {i+1}',
            'description': f'A great fitness workout session {i+1}',
            'nn_category': 'Fitness',
            'video_category': 'Fitness',
            'views': 10000 + i*1000,
            'channel': 'FitnessChannel',
            'duration': '15:30',
            'thumbnail': f'https://example.com/thumbnail_fitness{i}.jpg',
        })
    
    # Add 3 videos with different categories
    videos.append({
        'id': 'music1',
        'title': 'Best Music Playlist',
        'description': 'A collection of great songs',
        'nn_category': 'Music',
        'video_category': 'Music',
        'views': 20000,
        'channel': 'MusicChannel',
        'duration': '1:05:30',
        'thumbnail': 'https://example.com/thumbnail_music.jpg',
    })
    
    videos.append({
        'id': 'education1',
        'title': 'Learn Programming',
        'description': 'Introduction to programming concepts',
        'nn_category': 'Education',
        'video_category': 'Education',
        'views': 15000,
        'channel': 'EduChannel',
        'duration': '45:00',
        'thumbnail': 'https://example.com/thumbnail_edu.jpg',
    })
    
    videos.append({
        'id': 'cooking1',
        'title': 'Easy Cooking Recipes',
        'description': 'Simple recipes for beginners',
        'nn_category': 'Cooking',
        'video_category': 'Cooking',
        'views': 18000,
        'channel': 'CookingChannel',
        'duration': '25:15',
        'thumbnail': 'https://example.com/thumbnail_cooking.jpg',
    })
    
    # Before diversity
    print("\nBefore ensuring diversity:")
    for i, video in enumerate(videos[:6]):
        category = video.get('nn_category', video.get('video_category', 'Unknown'))
        print(f"{i+1}. {video['title']} (Category: {category})")
    
    # Apply diversity
    diverse_videos = enhancer.ensure_content_diversity(videos.copy())
    
    # After diversity
    print("\nAfter ensuring diversity:")
    for i, video in enumerate(diverse_videos[:6]):
        category = video.get('nn_category', video.get('video_category', 'Unknown'))
        print(f"{i+1}. {video['title']} (Category: {category})")

if __name__ == "__main__":
    # Run tests
    test_time_of_day_scoring()
    test_collaborative_filtering()
    test_content_diversity()
    test_reranking()
    
    print("\nAll enhancement tests completed!")
