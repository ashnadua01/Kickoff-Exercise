from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import json
import os
from contextual_analysis import ContextAnalyzer
from emotion_detector import EmotionDetector
from config import EMOTION_MAPPING, RELATIONSHIP_CONTEXTS, PRIMARY_EMOTIONS, SECONDARY_EMOTIONS

# Initialize FastAPI app
app = FastAPI(
    title="Ki Relationship Assistant - Emotion Analysis API",
    description="API for detecting and analyzing emotions in relationship conversations",
    version="1.0.0"
)

# Initialize components
emotion_detector = EmotionDetector()
context_analyzer = ContextAnalyzer()

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="The text to analyze")
    user_id: Optional[str] = Field(None, description="ID of the user (optional)")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation (optional)")
    
class EmotionScore(BaseModel):
    emotion: str
    score: float
    is_primary: bool
    
class ContextScore(BaseModel):
    context: str
    score: float
    
class EmotionResponse(BaseModel):
    text: str
    emotions: List[EmotionScore]
    dominant_emotion: str
    contexts: List[ContextScore]
    dominant_context: Optional[str]
    confidence: float
    
class ConversationInput(BaseModel):
    user1_text: str
    user2_text: str
    conversation_id: Optional[str] = None

class ConversationAnalysisResponse(BaseModel):
    user1_analysis: EmotionResponse
    user2_analysis: EmotionResponse
    relationship_dynamics: Dict[str, float]
    potential_conflict_areas: List[str]
    communication_insights: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to Ki Relationship Assistant Emotion Analysis API"}

@app.post("/analyze/emotion", response_model=EmotionResponse)
async def analyze_emotion(input_data: TextInput):
    """
    Analyze emotions in a single text input
    """
    try:
        # Detect emotions
        emotions = emotion_detector.detect_emotions(input_data.text)
        
        # Analyze context
        contexts = context_analyzer.analyze_context(input_data.text, emotions)
        
        # Format response
        emotion_scores = []
        for emotion, score in emotions.items():
            is_primary = emotion in PRIMARY_EMOTIONS
            emotion_scores.append(EmotionScore(
                emotion=emotion,
                score=float(score),
                is_primary=is_primary
            ))
        
        # Sort by score
        emotion_scores.sort(key=lambda x: x.score, reverse=True)
        
        context_scores = []
        for context, score in contexts.items():
            context_scores.append(ContextScore(
                context=context,
                score=float(score)
            ))
        
        # Sort by score
        context_scores.sort(key=lambda x: x.score, reverse=True)
        
        dominant_emotion = emotion_scores[0].emotion if emotion_scores else None
        dominant_context = context_scores[0].context if context_scores else None
        
        # Calculate overall confidence
        confidence = max([e.score for e in emotion_scores]) if emotion_scores else 0.0
        
        return EmotionResponse(
            text=input_data.text,
            emotions=emotion_scores,
            dominant_emotion=dominant_emotion,
            contexts=context_scores,
            dominant_context=dominant_context,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/conversation", response_model=ConversationAnalysisResponse)
async def analyze_conversation(input_data: ConversationInput):
    """
    Analyze emotions in a conversation between two users
    """
    try:
        # Analyze individual texts
        user1_analysis = await analyze_emotion(TextInput(text=input_data.user1_text))
        user2_analysis = await analyze_emotion(TextInput(text=input_data.user2_text))
        
        # Analyze relationship dynamics
        relationship_dynamics = {}
        potential_conflicts = []
        communication_insights = []
        
        # Look for emotional mismatches
        u1_emotion = user1_analysis.dominant_emotion
        u2_emotion = user2_analysis.dominant_emotion
        
        # Calculate emotional alignment
        emotional_alignment = calculate_emotional_alignment(
            user1_analysis.emotions, 
            user2_analysis.emotions
        )
        relationship_dynamics["emotional_alignment"] = emotional_alignment
        
        # Identify potential conflicts based on emotion combinations
        if (u1_emotion in ["anger", "frustration"] and 
            u2_emotion in ["defensiveness", "withdrawal"]):
            potential_conflicts.append("Criticism-Defensiveness Pattern")
            communication_insights.append(
                "Try using 'I feel' statements instead of criticism or blame"
            )
            
        if (u1_emotion in ["sadness", "fear", "anxiety"] and 
            u2_emotion in ["dismissiveness", "neutrality"]):
            potential_conflicts.append("Emotional Invalidation")
            communication_insights.append(
                "Practice acknowledging each other's emotions even if you don't fully understand them"
            )
        
        # Look for context mismatches
        u1_context = user1_analysis.dominant_context
        u2_context = user2_analysis.dominant_context
        
        if u1_context != u2_context:
            potential_conflicts.append(f"Different focus areas: {u1_context} vs {u2_context}")
            communication_insights.append(
                "You may be talking about different issues - try to align on what the core concern is"
            )
        
        # Default insights if none generated
        if not communication_insights:
            communication_insights.append(
                "Continue open communication while being mindful of each other's emotions"
            )
            
        return ConversationAnalysisResponse(
            user1_analysis=user1_analysis,
            user2_analysis=user2_analysis,
            relationship_dynamics=relationship_dynamics,
            potential_conflict_areas=potential_conflicts,
            communication_insights=communication_insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation analysis failed: {str(e)}")

def calculate_emotional_alignment(emotions1, emotions2):
    """Calculate how aligned two users' emotions are"""
    # Simple implementation - can be enhanced
    if not emotions1 or not emotions2:
        return 0.5
    
    # Get the top 3 emotions from each user
    top_emotions1 = {e.emotion: e.score for e in emotions1[:3]}
    top_emotions2 = {e.emotion: e.score for e in emotions2[:3]}
    
    # Calculate overlap
    common_emotions = set(top_emotions1.keys()).intersection(set(top_emotions2.keys()))
    
    # If there are common emotions, calculate similarity
    if common_emotions:
        similarity = sum(min(top_emotions1[e], top_emotions2[e]) for e in common_emotions)
        return min(1.0, similarity)
    else:
        # Calculate distance between emotion vectors
        all_emotions = set(list(top_emotions1.keys()) + list(top_emotions2.keys()))
        vec1 = [top_emotions1.get(e, 0.0) for e in all_emotions]
        vec2 = [top_emotions2.get(e, 0.0) for e in all_emotions]
        
        # Cosine similarity
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = sum(a*a for a in vec1) ** 0.5
        norm2 = sum(b*b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

@app.get("/info/emotions")
async def list_emotions():
    """
    Return the list of emotions the system can detect
    """
    return {
        "primary_emotions": PRIMARY_EMOTIONS,
        "secondary_emotions": SECONDARY_EMOTIONS
    }

@app.get("/info/contexts")
async def list_contexts():
    """
    Return the list of relationship contexts the system can identify
    """
    return {
        "relationship_contexts": RELATIONSHIP_CONTEXTS
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)