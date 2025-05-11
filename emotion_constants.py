"""
Emotion constants module for the Emotion-Based Video Recommendation system.
Contains emotion mappings and descriptions used for recommendations.
"""

# Emotion to video mapping
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

# Emotion descriptions for Gemini context
EMOTION_DESCRIPTIONS = {
    "happy": "You're in a happy mood. You might enjoy content that amplifies your positive feelings, makes you laugh, or celebrates joyful moments.",
    "sad": "You seem to be feeling sad. You might benefit from content that provides comfort, gentle uplift, or helps process emotions.",
    "angry": "You appear to be feeling angry. You might benefit from content that helps you calm down, relax, or redirect your focus.",
    "fear": "You seem to be experiencing fear or anxiety. You might benefit from content that provides reassurance, calm, or positive distraction.",
    "surprise": "You look surprised. You might enjoy content that further stimulates your curiosity or shows you more amazing things.",
    "neutral": "You appear to be in a neutral mood. You might enjoy informative content that engages your mind.",
    "disgust": "You seem to be feeling disgusted. You might benefit from content that provides pleasant visual relief or positive sensory experiences."
}
