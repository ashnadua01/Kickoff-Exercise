import re
from config import RELATIONSHIP_CONTEXTS

class ContextAnalyzer:
    """
    Class responsible for analyzing the context of conversations in relationships
    """
    
    def __init__(self):
        """
        Initialize the context analyzer with predefined context patterns
        """
        # Initialize context patterns
        self.context_patterns = {
            "work_stress": r'\b(work|job|boss|deadline|career|promotion|office|workplace|colleague|coworker)\b',
            "financial_concerns": r'\b(money|finances?|debt|budget|spend|expensive|afford|cost|pay|bill|saving|income|salary)\b',
            "quality_time": r'\b(time together|quality time|date night|attention|neglect|busy|prioritize|schedule|make time)\b',
            "household_responsibilities": r'\b(chores?|cleaning|cooking|laundry|dishes|responsibility|task|housework|home|tidy)\b',
            "communication_issues": r'\b(talk|listen|hear|understand|communicate|express|share|conversation|discuss|silent|ignore)\b',
            "intimacy": r'\b(intimacy|sex|physical|affection|touch|kiss|hug|love language|romance|connection|distant)\b',
            "future_plans": r'\b(future|plan|goal|dream|ambition|marriage|kids|children|family|move|relocate|house|settle)\b',
            "in_laws_family": r'\b(family|in-?law|parent|mother|father|relative|sibling|brother|sister)\b',
            "health_concerns": r'\b(health|sick|illness|doctor|hospital|mental health|therapy|anxiety|depression|medication)\b',
            "trust_issues": r'\b(trust|honest|lie|cheat|faithful|loyalty|suspect|doubt|believe|confidence|faith)\b',
            "parenting": r'\b(child|kid|parent|mother|father|son|daughter|baby|school|discipline|raise)\b',
            "personal_growth": r'\b(grow|develop|improve|learn|change|better|habits|goal|ambition|self|personal)\b'
        }
    
    def analyze_context(self, text, emotions=None):
        """
        Analyze the context of the given text
        
        Args:
            text (str): The input text
            emotions (dict, optional): Dictionary of emotion scores
            
        Returns:
            dict: Dictionary of context scores
        """
        context_scores = {}
        
        # Check each context pattern
        for context, pattern in self.context_patterns.items():
            matches = re.findall(pattern, text.lower())
            # Calculate a simple score based on number of matches
            score = min(1.0, len(matches) * 0.2)  # Cap at 1.0
            
            # If there are matches, add to context scores
            if score > 0:
                context_scores[context] = score
        
        # If emotions are provided, enhance context detection
        if emotions:
            self._enhance_context_with_emotions(context_scores, emotions)
        
        # If no contexts detected, use a default
        if not context_scores:
            context_scores["general_relationship"] = 1.0
        
        # Normalize scores
        total = sum(context_scores.values())
        if total > 0:
            context_scores = {k: v/total for k, v in context_scores.items()}
        
        return context_scores
    
    def _enhance_context_with_emotions(self, context_scores, emotions):
        """
        Enhance context detection using the detected emotions
        
        Args:
            context_scores (dict): Current context scores
            emotions (dict): Dictionary of emotion scores
            
        Returns:
            None (modifies context_scores in-place)
        """
        # Example emotion-context connections
        if "anger" in emotions and emotions["anger"] > 0.3:
            # Anger could increase likelihood of conflict contexts
            if "communication_issues" in context_scores:
                context_scores["communication_issues"] *= 1.2
            if "trust_issues" in context_scores:
                context_scores["trust_issues"] *= 1.2
        
        if "sadness" in emotions and emotions["sadness"] > 0.3:
            # Sadness could relate to emotional distance
            if "intimacy" in context_scores:
                context_scores["intimacy"] *= 1.2
            if "quality_time" in context_scores:
                context_scores["quality_time"] *= 1.2
        
        if "anxiety" in emotions and emotions["anxiety"] > 0.3:
            # Anxiety could relate to future uncertainty
            if "future_plans" in context_scores:
                context_scores["future_plans"] *= 1.2
            if "financial_concerns" in context_scores:
                context_scores["financial_concerns"] *= 1.2
        
        # Cap values at 1.0
        for context in context_scores:
            context_scores[context] = min(1.0, context_scores[context])