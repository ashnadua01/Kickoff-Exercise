# Mappings for emotions from various models to our standardized set
EMOTION_MAPPING = {
    # Google's GoEmotions taxonomy
    "admiration": "admiration",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "frustration",
    "approval": "satisfaction",
    "caring": "love",
    "confusion": "confusion",
    "curiosity": "interest",
    "desire": "desire",
    "disappointment": "disappointment",
    "disapproval": "disapproval",
    "disgust": "disgust",
    "embarrassment": "embarrassment",
    "excitement": "excitement",
    "fear": "fear",
    "gratitude": "gratitude",
    "grief": "grief",
    "joy": "joy",
    "love": "love",
    "nervousness": "anxiety",
    "optimism": "hope",
    "pride": "pride",
    "realization": "realization",
    "relief": "relief",
    "remorse": "guilt",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    
    # Additional mappings from other models
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "worried": "anxiety",
    "frustrated": "frustration",
    "hopeful": "hope",
    "guilty": "guilt",
    "ashamed": "shame",
    "proud": "pride",
    "grateful": "gratitude",
    "content": "contentment",
    "excited": "excitement",
    "stressed": "stress",
    "lonely": "loneliness",
    "jealous": "jealousy",
    "confused": "confusion",
    "hurt": "hurt",
    "defensive": "defensiveness",
    "dismissive": "dismissiveness",
    "contempt": "contempt"
}

# Primary emotions - fundamental emotional states
PRIMARY_EMOTIONS = [
    "anger",
    "fear",
    "joy", 
    "sadness",
    "surprise",
    "disgust",
    "love",
    "neutral"
]

# Secondary emotions - more nuanced emotional states
SECONDARY_EMOTIONS = [
    "frustration",
    "disappointment",
    "anxiety",
    "confusion",
    "hope",
    "guilt",
    "shame",
    "pride",
    "gratitude",
    "contentment",
    "excitement",
    "stress",
    "loneliness",
    "jealousy",
    "hurt",
    "defensiveness",
    "dismissiveness",
    "contempt",
    "vulnerability",
    "trust",
    "admiration",
    "satisfaction",
    "interest",
    "desire",
    "disapproval",
    "embarrassment",
    "grief",
    "realization",
    "relief"
]

# Relationship contexts
RELATIONSHIP_CONTEXTS = [
    "work_stress",
    "financial_concerns",
    "quality_time",
    "household_responsibilities",
    "communication_issues",
    "intimacy",
    "future_plans",
    "in_laws_family",
    "health_concerns",
    "trust_issues",
    "parenting",
    "personal_growth",
    "general_relationship"
]