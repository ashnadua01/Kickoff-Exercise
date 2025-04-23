# Ki Relationship Assistant - Emotion Analysis Component

This repository contains the emotion analysis component for the Ki relationship assistant app. 
The system analyzes emotions in user text, identifies relationship contexts, and provides insights for healthier communication.

## Architecture Overview
1. **EmotionDetector**: Identifies primary and secondary emotions from text input using transformer models.
3. **ContextAnalyzer**: Determines relationship-specific contexts from the text.
4. **FastAPI Application**: Provides API endpoints for analyzing emotions in individual texts and conversations.

## Features

- Detection of primary emotions (anger, sadness, fear, joy, and more)
- Detection of secondary/nuanced emotions (disappointment, frustration, vulnerability, and more)
- Context-aware analysis specific to relationships
- Confidence scoring for detected emotions
- Relationship dynamics analysis for conversations

## How to run
1. Clone this repository:
```
git clone https://github.com/ashnadua01/Kickoff-Exercise.git
cd ki-emotion-analysis
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
uvicorn app:app --reload
```
The API will be available at http://localhost:8000

## API Endpoints

### Analyze Individual Text
```
POST /analyze/emotion
```
Request body:
```json
{
  "text": "I'm feeling frustrated because you never listen to me when I talk about my day",
  "user_id": "user123",
  "conversation_id": "conv456"
}
```

### Analyze Conversation
```
POST /analyze/conversation
```
Request body:
```json
{
  "user1_text": "You never help with the housework. I'm tired of doing everything myself!",
  "user2_text": "I feel like no matter what I do, it's never enough for you.",
  "conversation_id": "conv456"
}
```

### Get Supported Emotions
```
GET /info/emotions
```

### Get Supported Contexts
```
GET /info/contexts
```

## Model Selection
1. **Primary Emotion Detection**: I have used `joeddav/distilbert-base-uncased-go-emotions-student` which is trained on Google's GoEmotions dataset. This model can identify 27 emotion categories with good accuracy.

2. **Secondary Emotion Detection**: I have used `bhadresh-savani/distilbert-base-uncased-emotion` which provides more nuanced emotion detection.

## Limitations and Future Improvements

### Current Limitations
1. **Hindi Language Support**: Due to unavailability of data / pre-trained models for detecting emotions in Hinglish/Hindi text, the Hindi Support feature could not be added. However, to add it, a language identifier file would need to be created which will first detect the language and then pass it to the corresponding model for emotion detection.
2. **Limited Cultural Context**: The emotion models are primarily trained on English data, which may not capture cultural nuances in expressing emotions.

### Future Improvements

1. **Fine-tuned Hindi Emotion Model**: Train a dedicated model on Hindi emotional text data.
2. **Fine-tuned Hinglish Emotion Model**: Train a dedicated model on Hinglish emotional text data.
3. **Personalization**: Adapt to individual users' communication styles and emotion expressions over time.
4. **Voice Input Processing**: Add support for speech-to-text and emotion detection from audio.

## Datasets Used

1. **GoEmotions**: A dataset of 58k English Reddit comments labeled with 27 emotion categories.
   - https://github.com/google-research/google-research/tree/master/goemotions
