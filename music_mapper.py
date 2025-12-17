# music_mapper.py

def get_music_recommendation(emotion):
    """
    Takes detected emotion as input
    Returns music recommendation details
    """

    emotion = emotion.lower()

    music_map = {
        "happy": {
            "mood": "Energetic",
            "keywords": [
                "energetic songs",
                "feel good music",
                "happy upbeat playlist"
            ],
            "message": "You're feeling happy! Enjoy some energetic music 🎶"
        },

        "sad": {
            "mood": "Calm",
            "keywords": [
                "calm relaxing music",
                "soothing songs",
                "peaceful piano music"
            ],
            "message": "Feeling low? Here's some calm music to relax 🌿"
        },

        "angry": {
            "mood": "Peaceful",
            "keywords": [
                "peaceful instrumental music",
                "nature sounds",
                "meditation music"
            ],
            "message": "Let's cool down with peaceful music 🧘"
        },

        "neutral": {
            "mood": "Focus",
            "keywords": [
                "lofi beats",
                "study music",
                "ambient background music"
            ],
            "message": "Stay focused with some background music 🎧"
        }
    }

    return music_map.get(
        emotion,
        {
            "mood": "General",
            "keywords": ["popular music playlist"],
            "message": "Enjoy some music 🎵"
        }
    )
