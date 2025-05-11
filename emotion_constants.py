"""
Emotion constants module for the Emotion-Based Video Recommendation system.
Contains emotion mappings and descriptions used for recommendations.
"""

# Emotion to video mapping
EMOTION_VIDEO_MAPPING = {
    "happy": {
        "keywords": ["learning techniques", "growth mindset", "financial literacy", "startup success stories", "business strategies", "holiday planning"],
        "categories": ["27", "28", "10"],  # Education, Science & Tech, How-To & DIY
    },
    "sad": {
        "keywords": ["motivational speeches", "standup comedy", "uplifting music", "inspiring stories"],
        "categories": ["23", "22", "24"],  # Comedy, Music, Entertainment
    },
    "angry": {
        "keywords": ["calming music", "positive facts", "relaxation techniques", "peaceful content"],
        "categories": ["22", "27", "26"],  # Music, Education, How-To & Style
    },
    "fear": {
        "keywords": ["guided meditation", "calming music", "relaxation techniques", "soothing tones"],
        "categories": ["22", "26"],  # Music, How-To & Style
    },
    "surprise": {
        "keywords": ["fascinating facts", "new discoveries", "educational content", "cutting-edge research"],
        "categories": ["28", "27", "24"],  # Science & Tech, Education, Entertainment
    },
    "neutral": {
        "keywords": ["learning techniques", "growth mindset", "financial education", "business strategies"],
        "categories": ["27", "28", "10"],  # Education, Science & Tech, How-To & DIY
    },
    "disgust": {
        "keywords": ["meditation techniques", "mindfulness practices", "calming visuals", "peaceful content"],
        "categories": ["26", "22"],  # How-To & Style, Music
    }
}

# Emotion descriptions for Gemini context
EMOTION_DESCRIPTIONS = {
    "happy": "You're in a happy mood. This is a great time for learning new things, exploring growth mindset concepts, or diving into topics like finance, startups, or business. Maybe even plan your next holiday!",
    "sad": "You seem to be feeling down. Some motivational content, standup comedy, or uplifting music might help improve your mood.",
    "angry": "You appear to be feeling angry. Calming music and positive facts can help soothe your mind and redirect your focus in a productive way.",
    "fear": "You seem to be experiencing anxiety. Meditation, soothing music, and calming content can help restore your sense of peace and security.",
    "surprise": "You look surprised! This is a perfect time to learn about new discoveries and fascinating facts that will expand your knowledge.",
    "neutral": "You appear to be in a balanced mood. This is a great opportunity to explore learning resources, growth mindset concepts, or business and financial topics.",
    "disgust": "You seem to be feeling uncomfortable. Meditation practices can help center your mind and provide a more peaceful perspective."
}
