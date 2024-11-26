from openai import OpenAI
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class OpenRouterComparator:
    def __init__(self):
        """Initialize the OpenRouterComparator."""
        self._api_key = os.environ.get('OPENROUTER_API_KEY')
        self._client = None
        self._cached_models = None
        
        # Initialize client if API key is available
        if self._api_key:
            try:
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self._api_key,
                    default_headers={
                        "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter",
                        "X-Title": "AssignmentGraderPro"
                    }
                )
            except Exception as e:
                print(f"Error initializing OpenRouter client: {str(e)}")
        
        self.available_models = self.get_available_models()

    def set_api_key(self, api_key: str) -> List[str]:
        """Set the OpenRouter API key and initialize client."""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key provided")
        
        try:
            self._api_key = api_key
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter",
                    "X-Title": "AssignmentGraderPro"
                }
            )
            
            # Update available models
            self.available_models = self.get_available_models()
            return self.available_models
            
        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "invalid" in error_msg:
                raise ValueError("Invalid API key")
            else:
                raise ValueError(f"Error connecting to OpenRouter: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models."""
        if self._cached_models is not None:
            return self._cached_models

        # Default models if API is not available
        default_models = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4-turbo-preview",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "google/gemini-pro",
            "meta/llama2-70b-chat",
            "mistral/mistral-large"
        ]

        if not self._api_key or not self._client:
            return default_models

        try:
            # Fetch models from API
            response = self._client.models.list()
            available_models = []

            if hasattr(response, 'data'):
                for model in response.data:
                    if hasattr(model, 'id'):
                        available_models.append(model.id)

            # Cache the results
            self._cached_models = sorted(available_models) if available_models else default_models
            return self._cached_models

        except Exception as e:
            print(f"Error fetching models from OpenRouter API: {str(e)}")
            return default_models

    def compare_texts(self, student_text: str, solution_text: str, 
                     model_name: str = "openai/gpt-3.5-turbo",
                     marking_criteria: str = None,
                     temperature: float = 0.1,
                     top_p: float = 1.0,
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Compare student text with solution text using OpenRouter."""
        if not self._client:
            return {
                "detailed_feedback": "Error: OpenRouter API key not set. Please provide your API key in the settings."
            }

        if not isinstance(model_name, str) or model_name not in self.available_models:
            print(f"Invalid model name '{model_name}', falling back to {self.available_models[0]}")
            model_name = self.available_models[0]

        try:
            criteria_section = ""
            if marking_criteria and marking_criteria.strip():
                criteria_section = f"Use these marking criteria: {marking_criteria}"

            prompt = f'''Grade the student submission by comparing it with the solution. Return a JSON object with scores and feedback for each question and a total score. Format your response EXACTLY as:

{{"feedback": "Question 1: [Score: X/Y] <feedback>
Question 2: [Score: X/Y] <feedback>
Question 3: [Score: X/Y] <feedback>
Total Score: X/20"}}

Solution:
{solution_text}

Student submission:
{student_text}

{criteria_section}'''

            # Get response from OpenRouter
            chat_completion = self._client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise grading assistant. Always respond with valid JSON objects only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "transforms": ["middle-out"]
                }
            )

            # Extract and parse response
            if not chat_completion or not hasattr(chat_completion, 'choices') or not chat_completion.choices:
                return {
                    "detailed_feedback": "Error: No response from OpenRouter API. Please try again."
                }
            
            response_text = chat_completion.choices[0].message.content
            if not response_text:
                return {
                    "detailed_feedback": "Error: Empty response from OpenRouter API. Please try again."
                }
            
            try:
                # Parse response
                response_data = json.loads(response_text)
                
                if isinstance(response_data, dict):
                    if 'feedback' in response_data:
                        return {'detailed_feedback': response_data['feedback']}
                    
                    parts = []
                    for key, value in response_data.items():
                        if isinstance(value, dict):
                            score = value.get('score', value.get('gained_mark', value.get('marks', '')))
                            feedback = value.get('feedback', '')
                            parts.append(f"{key}:")
                            if score:
                                parts.append(f"Score: {score}")
                            if feedback:
                                parts.append(feedback)
                        else:
                            parts.append(f"{key}: {value}")
                    
                    return {'detailed_feedback': '\n'.join(parts)}
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                return {'detailed_feedback': response_text}
                
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                return {
                    "detailed_feedback": "Error: Request timed out. Please try again."
                }
            elif "404" in error_msg:
                return {
                    "detailed_feedback": f"Error: Invalid model '{model_name}'. Please select a different model."
                }
            elif "rate" in error_msg:
                return {
                    "detailed_feedback": "Error: Rate limit exceeded. Please try again later."
                }
            return {
                "detailed_feedback": f"Error processing submission: {error_msg}"
            }
