import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from config import EMOTION_MAPPING, PRIMARY_EMOTIONS, SECONDARY_EMOTIONS
import re

class EmotionDetector:
    """
    Class responsible for detecting emotions in text across multiple languages
    """
    
    def __init__(self):
        # Load English emotion detection model
        self.en_model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
        self.en_tokenizer = AutoTokenizer.from_pretrained(self.en_model_name)
        self.en_model = AutoModelForSequenceClassification.from_pretrained(self.en_model_name)
        self.en_emotion_classifier = pipeline(
            "text-classification", 
            model=self.en_model, 
            tokenizer=self.en_tokenizer,
            return_all_scores=True
        )
        
        # Load secondary emotion detector (fine-grained emotions)
        self.secondary_emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.secondary_emotion_tokenizer = AutoTokenizer.from_pretrained(self.secondary_emotion_model_name)
        self.secondary_emotion_model = AutoModelForSequenceClassification.from_pretrained(self.secondary_emotion_model_name)
        self.secondary_emotion_classifier = pipeline(
            "text-classification",
            model=self.secondary_emotion_model,
            tokenizer=self.secondary_emotion_tokenizer,
            return_all_scores=True
        )
    
    def detect_emotions(self, text):
        """
        Detect emotions in the given text
        
        Args:
            text (str): The input text
            
        Returns:
            dict: Dictionary of emotion scores
        """

        primary_results = self.en_emotion_classifier(text)
        secondary_results = self.secondary_emotion_classifier(text)

        emotion_scores = {}

        for emotion_set in primary_results:
            for emotion in emotion_set:
                if emotion['label'] in EMOTION_MAPPING:
                    mapped_label = EMOTION_MAPPING[emotion['label']]
                    # Only include if it's a primary emotion or score is significant
                    if mapped_label in PRIMARY_EMOTIONS or emotion['score'] > 0.1:
                        emotion_scores[mapped_label] = emotion['score']
        
        # Process secondary emotions (fine-grained model)
        for emotion_set in secondary_results:
            for emotion in emotion_set:
                # Map the emotion labels
                if emotion['label'] in SECONDARY_EMOTIONS:
                    emotion_scores[emotion['label']] = emotion['score']
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores