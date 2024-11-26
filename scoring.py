from scoring_criteria import ScoringCriteria
import re

class ScoreCalculator:
    def __init__(self):
        self.criteria = ScoringCriteria()
    
    def update_scoring_criteria(self, new_criteria):
        """Update the scoring criteria."""
        self.criteria.update_criteria(new_criteria)
    
    def reset_scoring_criteria(self):
        """Reset scoring criteria to default."""
        self.criteria.reset_to_default()
    
    def extract_score(self, detailed_feedback):
        """Extract score from detailed feedback using comprehensive pattern matching."""
        try:
            feedback_text = str(detailed_feedback)
            
            # 1. First look for explicit final scores with various formats
            final_score_patterns = [
                # Standard formats
                r'(?:Total|Final|Overall)\s+(?:Score|Mark|Grade|Points|Marks):\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                r'(?:Total|Final|Overall)\s+(?:Score|Mark|Grade|Points|Marks)\s*(?:for|=|is)?\s*(?:the\s+whole\s+solution|all\s+questions)?:\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                r'Final\s+Mark\s+for\s+the\s+Whole\s+Solution\s*(?:\(.*?\))?\s*:\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                r'Total\s+gained\s+marks?\s*:\s*(\d+(?:\.\d+)?)',
                
                # Variations with different separators
                r'(?:Score|Mark|Grade|Points):\s*(\d+(?:\.\d+)?)/20',
                r'(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20\s*(?:points|marks|total)?',
                
                # Percentage formats
                r'(?:Final|Total|Overall)\s+Percentage:\s*(\d+(?:\.\d+)?)\s*%',
                r'Grade:\s*(\d+(?:\.\d+)?)\s*%',
                
                # JSON-like formats
                r'["\']total_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']final_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']marks["\']?\s*:\s*(\d+(?:\.\d+)?)',
                
                # Bullet point or list formats
                r'•\s*(?:Total|Final|Overall)\s+Score:\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                r'[-*]\s*(?:Total|Final|Overall)\s+Score:\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                
                # Parenthetical formats
                r'Total\s*\((\d+(?:\.\d+)?)/20\)',
                r'\((\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20\s*(?:total)?\)',
            ]
            
            for pattern in final_score_patterns:
                match = re.search(pattern, feedback_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    score = float(match.group(1))
                    if score <= 100 and 'percentage' in pattern.lower():
                        score = (score / 100) * 20
                    return round(max(0, min(20, score)), 1)
            
            # 2. Look for question-by-question scores
            question_patterns = [
                # Individual question total marks
                r'Question\s*\d+.*?Total\s*Marks:\s*(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*\d+',
                r'Question\s*\d+\s*:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                r'Q\d+\s*:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                
                # Section scores
                r'Part\s*\([a-z]\).*?:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                r'Section\s*\d+.*?:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                
                # Score keywords
                r'(?:score|marks|points|grade):\s*(\d+(?:\.\d+)?)',
                r'awarded:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                r'earned:\s*(\d+(?:\.\d+)?)\s*(?:marks|points)',
                
                # Bullet points or lists
                r'•\s*Score:\s*(\d+(?:\.\d+)?)',
                r'[-*]\s*Marks:\s*(\d+(?:\.\d+)?)',
            ]
            
            all_scores = []
            for pattern in question_patterns:
                scores = re.findall(pattern, feedback_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                all_scores.extend([float(score) for score in scores if score])
            
            if all_scores:
                # Strategy 1: If we find a score that's close to 20, it's likely the total
                for score in sorted(all_scores, reverse=True):
                    if 15 <= score <= 20:  # Reasonable range for final score
                        return round(score, 1)
                
                # Strategy 2: Look for the last score mentioned (often the total)
                last_score = float(all_scores[-1])
                if last_score <= 20:
                    return round(last_score, 1)
                elif last_score <= 100:  # Percentage
                    return round((last_score / 100) * 20, 1)
                
                # Strategy 3: Sum up scores if they're all small
                if all(score <= 10 for score in all_scores):
                    total = sum(all_scores)
                    if total <= 20:
                        return round(total, 1)
            
            # 3. Last resort: Look for any numbers followed by "out of 20" or "/20"
            final_attempts = [
                r'(\d+(?:\.\d+)?)\s*(?:out\s*of|/)\s*20',
                r'20\s*:\s*(\d+(?:\.\d+)?)',
                r'(?:marks|points|score):\s*(\d+(?:\.\d+)?)',
            ]
            
            for pattern in final_attempts:
                matches = re.findall(pattern, feedback_text, re.IGNORECASE)
                if matches:
                    scores = [float(score) for score in matches if score]
                    if scores:
                        # Take the last occurrence as it's often the final score
                        score = scores[-1]
                        if score <= 20:
                            return round(score, 1)
                        elif score <= 100:  # Handle percentage
                            return round((score / 100) * 20, 1)
            
            return 0.0
                
        except Exception as e:
            print(f"Error extracting score: {str(e)}")
            return 0.0
    
    def calculate_score(self, detailed_feedback):
        """Calculate score from detailed feedback (legacy method)."""
        return self.extract_score(detailed_feedback)
    
    def generate_feedback(self, detailed_feedback, final_score):
        """Generate detailed feedback based on comparison results and current criteria."""
        criteria = self.criteria.get_criteria()
        feedback = []
        
        # Use the AI-generated detailed feedback
        feedback.append(detailed_feedback)
        
        # Add score-based feedback
        if final_score >= 18:
            feedback.append("\nOutstanding performance!")
        elif final_score >= 15:
            feedback.append("\nVery good understanding shown.")
        elif final_score >= 10:
            feedback.append("\nSatisfactory work, but review the material.")
        else:
            feedback.append("\nSignificant revision needed. Please review the material.")
        
        return "\n".join(feedback)
