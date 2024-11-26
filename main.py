import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
from file_processor import FileProcessor
from groq_comparator import GroqTextComparator
from ollama_comparator import OllamaTextComparator
from openai_comparator import OpenAIComparator
from openai_compatible_comparator import OpenAICompatibleComparator
from mistral_comparator import MistralComparator
from openrouter_comparator import OpenRouterComparator
from scoring import ScoreCalculator
from report_generator import ReportGenerator
from statistical_analyzer import StatisticalAnalyzer
import io
import zipfile
import os
import re


def show_evaluation_settings(groq_comparator, ollama_comparator,
                           openai_compatible_comparator, openai_comparator,
                           mistral_comparator, openrouter_comparator):
    """Configure evaluation settings including LLM provider and marking criteria."""
    st.sidebar.markdown("### Evaluation Settings")

    # Initialize session state for model selection and API keys
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = groq_comparator.available_models[0]
    if 'selected_ollama_model' not in st.session_state:
        st.session_state.selected_ollama_model = ollama_comparator.available_models[0]
    if 'selected_openai_model' not in st.session_state:
        st.session_state.selected_openai_model = openai_compatible_comparator.available_models[0]
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'mistral_api_key' not in st.session_state:
        st.session_state.mistral_api_key = ""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', '')
    if 'openrouter_api_key' not in st.session_state:
        st.session_state.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if 'selected_cloud_openai_model' not in st.session_state:
        st.session_state.selected_cloud_openai_model = openai_comparator.available_models[0]
    if 'selected_mistral_model' not in st.session_state:
        st.session_state.selected_mistral_model = mistral_comparator.available_models[0]
    if 'selected_openrouter_model' not in st.session_state:
        st.session_state.selected_openrouter_model = openrouter_comparator.available_models[0]
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.8
    if 'p_sampling' not in st.session_state:
        st.session_state.p_sampling = 1.0
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 4000
    if 'openai_models' not in st.session_state:
        st.session_state.openai_models = openai_comparator.available_models
    if 'mistral_models' not in st.session_state:
        st.session_state.mistral_models = mistral_comparator.available_models
    if 'groq_models' not in st.session_state:
        st.session_state.groq_models = groq_comparator.available_models
    if 'openrouter_models' not in st.session_state:
        st.session_state.openrouter_models = openrouter_comparator.available_models
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {}
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = {
            'boxplot': go.Figure(),  # Initialize empty figure
            'distribution': go.Figure(),
            'performance': go.Figure(),
            'bands': {}
        }
    if 'marking_criteria' not in st.session_state:
        st.session_state.marking_criteria = """Marking Scheme for Question 1:
Total Points: 8

Part (a): Determining the x, y, and z Scalar Components of Vector F
Total: 3 Points

1. Calculation of Fx Component
Total: 1 Point
Marking:
- Award 1 point for correct the x-component: Fx 
- Deduct 0.5 points if there is an arithmetic error or if an incorrect angle is used or wrong calculation setup.
- full deduction (1point) if both calculation setup and final result are incorrect.

2. Calculation of Fy Component
Total: 1 Point
Marking:
- Award 1 point for correct the y-component: Fy 
- Deduct 0.5 points if there is an arithmetic error or if an incorrect angle is used or wrong calculation setup.
- full deduction (1point) if both calculation setup and final result are incorrect.

3. Calculation of Fz Component
Total: 1 Point
Marking:
- Award 1 point for correct the z-component: Fz 
- Deduct 0.5 points if there is an arithmetic error or if an incorrect angle is used or wrong calculation setup.
- full deduction (1point) if both calculation setup and final result are incorrect.

Part (b): Expressing F in Cartesian Vector Form
Total: 1 Point

1. Cartesian Vector Form Representation
Total: 1 Point
Marking:
- Award 1 point for correctly expressing F in the form: F = Fx * i + Fy * j + Fz * k
- Deduct 0.5 points if any component of the vector form is incorrect or missing.
- Deduct 0.25 points if there is an incorrect notation or sign error.

Part (c): Calculating Direction Angles omega, epsilon, and vartheta and Verifying the Condition
Total: 3 Points

1. Determination of Direction Angles α and β
Total: 2 Points
Marking:
- Award 1 point for each correctly calculated angle (α and β) using the formulas: cos(α) = Fx / |F|, cos(β) = Fy / |F|
- Deduct 0.5 points for any incorrect angle calculation or incorrect formula.

2. Verification of Direction Angle Requirement
Total: 1 Point
Marking:
- Award 1 point for correctly verifying that: cos^2(α) + cos^2(β) + cos^2(γ) = 1
- Deduct 0.5 points if there is an error in summing the squares of the cosines 
- full deduction (1 point) if this verification is missing

Part (d): Expressing F as a Product of Its Magnitude and Unit Vector
Total: 1 Point

1. Product of Magnitude and Unit Vector Expression
Total: 1 Point
Marking:
- Award 1 point for correctly expressing F in the form F = |F| * U_F, where U_F is the unit vector determined by: U_F = cos(α) * i + cos(β) * j + cos(γ) * k
- Deduct 0.5 points if any component of U_F is incorrect or if the vector is not properly normalized.

Final Calculation: Total Marks for Question 1
Total Points for Part (a) + Part (b) + Part (c) + Part (d)
Total Marks: [Total points] out of 8

Additional Notes:
	Emphasis on Calculation Process: Marks are awarded based on the process and setup of the formulas rather than specific numerical values.
	Full Marking for Correct Formula Usage: Full points awarded even if the final answer has minor rounding errors.
	5% Tolerance: A 5% tolerance is applied to all calculations, meaning that answers within 5% of the correct value will be considered correct.
	Single Critical Error: If a student makes a single mistake in the first step that affects all subsequent calculations, resulting in incorrect final answers, but all procedures are otherwise correct, the final grade for the question should not be less than 75% (6 points out of 8). This is to ensure that the student is not unfairly penalized for a single mistake that has a cascading effect on the rest of the problem.

Marking Scheme for Question 2:
Total Points: 4

Part (a): Calculating the Magnitude of Force F
Total: 0.5 Points
1. Correct Formula for Magnitude of F
Total: 0.5 Points
Marking:
- Award 0.5 points for using the correct formula to calculate the magnitude of F: |F| = sqrt(F_x^2 + F_y^2 + F_z^2) and for obtaining the correct magnitude of F.
- Deduct 0.25 points if there is an arithmetic or setup error in the magnitude calculation.
- Full deduction (0.5 points) if both the calculation setup and magnitude value are incorrect.

Part (b): Determining the Unit Vector U_F of Force F
Total: 0.5 Points

1. Correct Calculation of Unit Vector U_F
Total: 0.5 Points
Marking:
- Award 0.5 points for correctly calculating U_F as: 
U_F = F / |F| = (F_x / |F|) i + (F_y / |F|) j + (F_z / |F|) k
- Deduct 0.25 points if the calculation of U_F components is incorrect or incomplete.
- Full deduction (0.5 points) if both the calculation setup and U_F are incorrect.

Part (c): Determining the Position Vector r_A, Unit Vector U_r_A, and Equating Components to Solve for x, y, and L
Total: 3 Points

1. Formulation of Position Vector r_A
Total: 0.5 Points
Marking:
- Award 0.5 points for correctly setting up r_A with components: r_A = (x)i + (y)j + (5.5)k
- Full deduction (0.5 points) if the setup is incorrect.

2. Formulation of Unit Vector U_r_A
Total: 0.5 Points
Marking:
- Award 0.5 points for correctly expressing U_r_A as:
 U_r_A = r_A / |r_A| = ((x)i + (y)j + (5.5)k) / L
- Full deduction (0.5 points) if the setup or expression of U_r_A is incorrect.

3. Equating Unit Vector Components to Find L
Total: 1 Point
Marking:
- Award 1 point for correctly using the k-component to solve for L: L = 5.5 / U_F_z
- Deduct 0.5 points if the value of L is incorrect but the calculation setup is correct.
- Full deduction (1 point) if both the result for L and the calculation setup are incorrect.

4. Calculating Coordinates x and y
Total: 1 Point
Marking:
- Award 0.5 points each for correctly calculating x and y by equating the i and j components: x = U_F_x * L, y = U_F_y * L
- Full deduction (1 point) if the calculation or setup for x and y is incorrect.

Final Calculation:
Total Marks for Question 2: Total Points for Part (a) + Part (b) + Part (c)
Total Marks: [Total points] out of 4

Additional Notes:
	Emphasis on Calculation Process: Marks are awarded based on the process and setup of the formulas rather than specific numerical values
	Clarity of Unit Vector Verification: Full marks awarded even if minor rounding errors occur in U_F.
	5% Tolerance: A 5% tolerance is applied to all calculations, meaning that answers within 5% of the correct value will be considered correct.
	Single Critical Error: If a student makes a single error in the first step that affects all subsequent calculations, resulting in incorrect final answers, but all other procedures are correct, ensure the final grade for the question is not less than 75% (3 points out of 4) to prevent over-penalizing.

Marking Scheme for Question 3:
Total Points: 8

Part (a): Express Each Cable Force in Cartesian Vector Form
Total: 3 Points

1. Expression of Force F_A
Total: 1 Point
Marking:
- Award 1 point for correctly expressing F_A in Cartesian vector form.
- Deduct 0.5 points for arithmetic or setup errors.
- Full deduction (1 point) if both the answer and calculation steps are wrong.

2. Expression of Force F_B
Total: 1 Point
Marking:
- Award 1 point for correctly expressing F_B in Cartesian vector form.
- Deduct 0.5 points for arithmetic or setup errors.
- Full deduction (1 point) if both the answer and calculation steps are wrong.

3. Expression of Force F_C
Total: 1 Point
Marking:
- Award 1 point for correctly expressing F_C in Cartesian vector form.
- Deduct 0.5 points for arithmetic or setup errors.
- Full deduction (1 point) if both the answer and calculation steps are wrong.

Part (b): Determine the Resultant Force in Cartesian Vector Form and Its Magnitude and Direction Angles
Total: 3 Points
1. Calculation of Resultant Force F_R in cartesian notation
Total: 1 Point
Marking:
- Award 1 point for correctly calculating the resultant force F_R = F_A + F_B + F_C.
- Deduct 0.5 points for arithmetic or setup errors in the calculation.
- Full deduction (1 point) if both the answer and calculation steps are wrong.

2. Magnitude of the Resultant Force
Total: 1 Point
Marking:
- Award 1 point for correctly calculating the magnitude |F_R| using the formula: 
|F_R| = sqrt((F_R_x)^2 + (F_R_y)^2 + (F_R_z)^2)
- Deduct 0.5 points for arithmetic or setup errors in the magnitude calculation.
- Full deduction (1 point) if both the answer and calculation steps are wrong.

3. Direction Angles
Total: 1.5 Points
Marking:
- Award 0.5 points for each correct direction angle (α, β, and γ) using the formulas: 
  α = cos^-1(F_R_x / |F_R|), β = cos^-1(F_R_y / |F_R|), γ = cos^-1(F_R_z / |F_R|)
- Deduct 0.25 points for each direction angle if the calculation steps are correct but the result is incorrect.
- Full deduction (0.5 points) for each direction angle if both the answer and calculation steps are wrong.

Part (c): Determine the Angle at D Formed by Cables DB and DC
Total: 2 Points
1. Calculation of Angle θ
Total: 2 Points
Marking:
- Award 2 points for correctly calculating the angle phi using the dot product method: r_DC * r_DB = |r_DC||r_DB| cos(θ)
- Deduct 1 point for any arithmetic errors in calculating r_DC * r_DB or in setting up the formula.
- Full deduction (2 points) if both the answer and calculation steps are wrong or if the angle phi is not determined.

Final Calculation:
Total Marks for Question 3: Total Points for Part (a) + Part (b) + Part (c)
Total Marks: [Total points] out of 8

Additional Notes:
	Emphasis on Calculation Process: Marks are awarded based on the process and setup of the formulas rather than specific numerical values 
	Clarity of Unit Vector Verification: Full marks awarded even if minor rounding errors occur in calculations.
	5% Tolerance: A 5% tolerance is applied to all calculations, meaning that answers within 5% of the correct value will be considered correct.
	Single Critical Error: If a single error in the first step affects all subsequent calculations, resulting in incorrect final answers, ensure the final grade for the question is not less than 75% (6 points out of 8) to avoid over-penalizing.

Final Mark for the Whole Solution (Questions 1, 2, and 3)
Total Marks: [Total points for Question 1] + [Total points for Question 2] + [Total points for Question 3]
Total Marks: [Total points] out of 20

Note: The total marks for each question are added together to give the final mark for the whole solution, out of a total of 20 points."""

    # LLM Provider Selection
    llm_provider = st.sidebar.radio(
        "Select LLM Provider", [
            "Groq (Cloud)", "OpenAI (Cloud)", "Mistral (Cloud)", "Ollama (Local)",
            "OpenAI-compatible Server", "OpenRouter (Cloud)"
        ],
        help="Choose the AI model provider for evaluation")

    # Function to handle API key validation and model fetching
    def validate_api_key(provider, api_key, comparator, models_key):
        status_placeholder = st.sidebar.empty()

        # Show loading state
        with status_placeholder:
            with st.spinner(f'Validating {provider} API key...'):
                try:
                    # Special handling for Groq which uses environment variables
                    if provider == 'Groq':
                        available_models = comparator.available_models
                        if available_models:
                            st.session_state[models_key] = available_models
                            st.session_state.api_status[provider] = {
                                'valid': True,
                                'message': f"✅ {provider} API key is valid"
                            }
                            return True
                        else:
                            st.session_state.api_status[provider] = {
                                'valid': False,
                                'message': "❌ Invalid or missing Groq API key"
                            }
                            return False
                    else:
                        # For other providers that use set_api_key
                        available_models = comparator.set_api_key(api_key)
                        st.session_state[models_key] = available_models
                        st.session_state.api_status[provider] = {
                            'valid': True,
                            'message': f"✅ {provider} API key set successfully"
                        }
                        return True
                except ValueError as e:
                    st.session_state.api_status[provider] = {
                        'valid': False,
                        'message': f"❌ {str(e)}"
                    }
                    # Reset to default models on error
                    st.session_state[models_key] = comparator.available_models
                    return False
                except Exception as e:
                    st.session_state.api_status[provider] = {
                        'valid': False,
                        'message': f"❌ Unexpected error: {str(e)}"
                    }
                    st.session_state[models_key] = comparator.available_models
                    return False

    # Model Selection based on provider
    if llm_provider == "Groq (Cloud)":
        # Display current API key status for Groq
        groq_status = st.sidebar.empty()
        if 'Groq' in st.session_state.api_status:
            groq_status.markdown(st.session_state.api_status['Groq']['message'])
        else:
            # Initial validation of environment variable
            is_valid = validate_api_key('Groq', None, groq_comparator, 'groq_models')
            if is_valid:
                groq_status.success("✅ Groq API key is valid")
            else:
                groq_status.warning("⚠️ Groq API key is not set in environment variables")

        # Model selection dropdown
        st.sidebar.selectbox("Select Groq Model",
                           st.session_state.groq_models,
                           key='selected_model',
                           help="Choose a Groq model for evaluation")

    elif llm_provider == "OpenAI (Cloud)":
        # API Key input with clear button
        api_key_col, clear_col = st.sidebar.columns([4, 1])
        with api_key_col:
            api_key = st.text_input("OpenAI API Key",
                                  type="password",
                                  value=st.session_state.openai_api_key,
                                  key="openai_api_key",
                                  help="Enter your OpenAI API key")
        with clear_col:
            if st.button("Clear", key="clear_openai"):
                st.session_state.openai_api_key = ""
                st.session_state.openai_models = openai_comparator.available_models
                if 'api_status' in st.session_state:
                    st.session_state.api_status.pop('OpenAI', None)
                st.rerun()

        # Update OpenAI comparator with API key and fetch models
        if api_key:
            is_valid = validate_api_key('OpenAI', api_key, openai_comparator,
                                      'openai_models')

            # Show status message
            if 'OpenAI' in st.session_state.api_status:
                st.sidebar.markdown(st.session_state.api_status['OpenAI']['message'])

            # Update model selection if needed
            if is_valid and st.session_state.selected_cloud_openai_model not in st.session_state.openai_models:
                st.session_state.selected_cloud_openai_model = st.session_state.openai_models[
                    0]

        # Model selection dropdown
        st.sidebar.selectbox("Select OpenAI Model",
                           st.session_state.openai_models,
                           key='selected_cloud_openai_model',
                           help="Choose an OpenAI model for evaluation")

    elif llm_provider == "Mistral (Cloud)":
        # API Key input with clear button
        api_key_col, clear_col = st.sidebar.columns([4, 1])
        with api_key_col:
            api_key = st.text_input("Mistral API Key",
                                  type="password",
                                  value=st.session_state.mistral_api_key,
                                  key="mistral_api_key",
                                  help="Enter your Mistral API key")
        with clear_col:
            if st.button("Clear", key="clear_mistral"):
                st.session_state.mistral_api_key = ""
                st.session_state.mistral_models = mistral_comparator.available_models
                if 'api_status' in st.session_state:
                    st.session_state.api_status.pop('Mistral', None)
                st.rerun()

        # Update Mistral comparator with API key and fetch models
        if api_key:
            is_valid = validate_api_key('Mistral', api_key, mistral_comparator,
                                      'mistral_models')

            # Show status message
            if 'Mistral' in st.session_state.api_status:
                st.sidebar.markdown(st.session_state.api_status['Mistral']['message'])

            # Update model selection if needed
            if is_valid and st.session_state.selected_mistral_model not in st.session_state.mistral_models:
                st.session_state.selected_mistral_model = st.session_state.mistral_models[
                    0]

        # Model selection dropdown
        st.sidebar.selectbox("Select Mistral Model",
                           st.session_state.mistral_models,
                           key='selected_mistral_model',
                           help="Choose a Mistral model for evaluation")

    elif llm_provider == "OpenRouter (Cloud)":
        # API Key input with clear button
        api_key_col, clear_col = st.sidebar.columns([4, 1])
        with api_key_col:
            api_key = st.text_input("OpenRouter API Key",
                                  type="password",
                                  value=st.session_state.openrouter_api_key,
                                  key="openrouter_api_key",
                                  help="Enter your OpenRouter API key")
        with clear_col:
            if st.button("Clear", key="clear_openrouter"):
                st.session_state.openrouter_api_key = ""
                st.session_state.openrouter_models = openrouter_comparator.available_models
                if 'api_status' in st.session_state:
                    st.session_state.api_status.pop('OpenRouter', None)
                st.rerun()

        # Update OpenRouter comparator with API key and fetch models
        if api_key:
            is_valid = validate_api_key('OpenRouter', api_key, openrouter_comparator,
                                      'openrouter_models')

            # Show status message
            if 'OpenRouter' in st.session_state.api_status:
                st.sidebar.markdown(st.session_state.api_status['OpenRouter']['message'])

            # Update model selection if needed
            if is_valid and st.session_state.selected_openrouter_model not in st.session_state.openrouter_models:
                st.session_state.selected_openrouter_model = st.session_state.openrouter_models[
                    0]

        # Model selection dropdown
        st.sidebar.selectbox("Select OpenRouter Model",
                           st.session_state.openrouter_models,
                           key='selected_openrouter_model',
                           help="Choose an OpenRouter model for evaluation")
    elif llm_provider == "Ollama (Local)":
        # Check Ollama connection status
        ollama_status = st.sidebar.empty()
        try:
            if not ollama_comparator._check_ollama_connection():
                ollama_status.warning(
                    "⚠️ Ollama service not running. Please install and start Ollama.")
            else:
                ollama_status.success("✅ Ollama service running")
        except Exception as e:
            ollama_status.warning(f"⚠️ Ollama service not running: {str(e)}")

        st.sidebar.selectbox("Select Ollama Model",
                           ollama_comparator.available_models,
                           key='selected_ollama_model',
                           help="Choose an Ollama model for evaluation")
    else:  # OpenAI-compatible Server
        # Check server connection status
        server_status = st.sidebar.empty()
        connection_status = openai_compatible_comparator.get_connection_status()
        if connection_status['available']:
            server_status.success("✅ OpenAI-compatible server running")
        else:
            server_status.warning(f"⚠️ {connection_status['error']}")

        st.sidebar.selectbox("Select Model",
                           openai_compatible_comparator.available_models,
                           key='selected_openai_model',
                           help="Choose a model from the OpenAI-compatible server")

    # Add model parameters controls
    st.sidebar.markdown("### Model Parameters")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness in the output. Lower values make the output more focused and deterministic."
    )
    
    p_sampling = st.sidebar.slider(
        "P Sampling",
        min_value=0.05,
        max_value=1.0,
        value=st.session_state.p_sampling,
        step=0.05,
        help="Top-p (nucleus) sampling. Lower values make the output more focused on higher probability tokens."
    )
    max_tokens = st.sidebar.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100,
        help="Maximum number of tokens in the model's response.")

    # Update session state
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens

    # Marking Criteria
    st.sidebar.markdown("### Marking Criteria")
    marking_criteria = st.sidebar.text_area(
        "Enter Marking Criteria",
        value=st.session_state.marking_criteria,
        help="Enter specific marking criteria for evaluation")

    return llm_provider, marking_criteria, temperature, p_sampling, max_tokens


def display_feedback(feedback, score, unique_id):
    """Display formatted feedback with score."""
    st.markdown(f"**Score:** {score:.1f}/20")
    st.text_area("Detailed Feedback",
                 value=feedback,
                 height=200,
                 disabled=True,
                 key=f"feedback_{unique_id}")


def generate_sample_results():
    """Generate sample results for testing the Statistical Analysis tab."""
    import numpy as np

    num_samples = 30  # Increased sample size for better distribution
    results = []

    # Generate random scores with a more realistic distribution
    # Using a truncated normal distribution centered around 14 (70%)
    scores = np.random.normal(14, 3, num_samples)
    scores = np.clip(scores, 0, 20)  # Ensure scores are within valid range

    for i, score in enumerate(scores, 1):
        results.append({
            'filename': f'student_{i}.pdf',
            'student_id': f'{1000 + i}',
            'score': float(score),
            'feedback': f'Sample feedback for student {i}\nQuestion 1: [Score: {score/3:.1f}/6.67] Good understanding shown.\nQuestion 2: [Score: {score/3:.1f}/6.67] Clear explanation provided.\nQuestion 3: [Score: {score/3:.1f}/6.67] Well-structured response.'
        })

    return results


def main():
    st.set_page_config(page_title="Assignment Grader Pro",
                       page_icon="📚",
                       layout="wide")

    # Initialize session state for results and stats
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = {
            'boxplot': go.Figure(),  # Initialize empty figure
            'distribution': go.Figure(),
            'performance': go.Figure(),
            'bands': {}
        }

    # Add logo and title in a horizontal layout
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("university-of-alberta-logo.png", width=200)
    with col2:
        st.title("📚 Assignment Grader Pro")

    # Initialize components
    file_processor = FileProcessor()
    groq_comparator = GroqTextComparator()
    ollama_comparator = OllamaTextComparator()
    openai_compatible_comparator = OpenAICompatibleComparator()
    openai_comparator = OpenAIComparator()
    mistral_comparator = MistralComparator()
    openrouter_comparator = OpenRouterComparator()
    score_calculator = ScoreCalculator()
    report_generator = ReportGenerator()
    statistical_analyzer = StatisticalAnalyzer()

    # Show evaluation settings
    llm_provider, marking_criteria, temperature, p_sampling, max_tokens = show_evaluation_settings(
        groq_comparator, ollama_comparator, openai_compatible_comparator,
        openai_comparator, mistral_comparator, openrouter_comparator)

    # Create tabs
    upload_tab, results_tab, stats_tab, reports_tab = st.tabs(
        ['📄 Upload Files', '📊 Results', '📈 Statistical Analysis', '📑 Reports'])

    # Upload Files tab
    with upload_tab:
        st.markdown("### Upload Assignment Files")
        col1, col2 = st.columns(2)

        with col1:
            solution_file = st.file_uploader("Solution PDF",
                                           type=['pdf'],
                                           help="Upload the solution/marking guide PDF")

        with col2:
            student_files = st.file_uploader(
                "Student Submissions",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload student PDFs (format: StudentID_Assignment.pdf)")

        if solution_file and student_files:
            if st.button("Start Evaluation", type="primary", use_container_width=True):
                try:
                    with st.spinner():
                        # Process solution file
                        solution_text = file_processor.extract_text(solution_file)

                        # Process and evaluate student files
                        results = []
                        progress_container = st.container()

                        with progress_container:
                            # Create columns for progress information
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                progress_info = st.empty()
                            
                            with col2:
                                time_text = st.empty()
                                speed_text = st.empty()

                            total_files = len(student_files)
                            start_time = time.time()
                            processed_files = []

                            for idx, student_file in enumerate(student_files, 1):
                                try:
                                    # Calculate progress and timing information
                                    progress = idx / total_files
                                    elapsed_time = time.time() - start_time
                                    files_per_second = idx / elapsed_time if elapsed_time > 0 else 0
                                    remaining_files = total_files - idx
                                    estimated_time = remaining_files / files_per_second if files_per_second > 0 else 0
                                    
                                    # Update progress bar with gradient color
                                    progress_color = f'rgba(46, 134, 193, {min(1.0, progress + 0.2)})'
                                    progress_html = f"""
                                    <style>
                                    .stProgress > div > div > div > div {{
                                        background-color: {progress_color};
                                        background-image: linear-gradient(to right, #2ECC71, #3498DB);
                                    }}
                                    </style>
                                    """
                                    st.markdown(progress_html, unsafe_allow_html=True)
                                    progress_bar.progress(progress)
                                    
                                    # Update status with colorful text
                                    status_html = f"""
                                    <div style="color: #2E86C1; font-weight: bold; font-size: 1.1em;">
                                        Processing: {student_file.name}
                                    </div>
                                    """
                                    status_text.markdown(status_html, unsafe_allow_html=True)
                                    
                                    # Show progress information
                                    progress_html = f"""
                                    <div style="color: #2C3E50;">
                                        File {idx} of {total_files} ({(progress * 100):.1f}% complete)
                                    </div>
                                    """
                                    progress_info.markdown(progress_html, unsafe_allow_html=True)
                                    
                                    # Show timing information
                                    time_html = f"""
                                    <div style="color: #7F8C8D;">
                                        Elapsed: {elapsed_time:.1f}s<br>
                                        Remaining: {estimated_time:.1f}s
                                    </div>
                                    """
                                    time_text.markdown(time_html, unsafe_allow_html=True)
                                    
                                    # Show processing speed
                                    speed_html = f"""
                                    <div style="color: #27AE60;">
                                        Speed: {files_per_second:.1f} files/s
                                    </div>
                                    """
                                    speed_text.markdown(speed_html, unsafe_allow_html=True)

                                    # Process file
                                    student_text = file_processor.extract_text(student_file)
                                    student_id = file_processor.extract_student_id(
                                        student_file.name)

                                    # Compare texts using selected provider
                                    comparison_result = None
                                    if llm_provider == "Groq (Cloud)":
                                        comparison_result = groq_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            p_sampling=p_sampling,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "OpenAI (Cloud)":
                                        if not st.session_state.openai_api_key:
                                            st.error(
                                                "Please provide your OpenAI API key in the settings")
                                            break
                                        comparison_result = openai_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_cloud_openai_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            p_sampling=p_sampling,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "Mistral (Cloud)":
                                        if not st.session_state.mistral_api_key:
                                            st.error(
                                                "Please provide your Mistral API key in the settings")
                                            break
                                        comparison_result = mistral_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_mistral_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            p_sampling=p_sampling,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "OpenRouter (Cloud)":
                                        if not st.session_state.openrouter_api_key:
                                            st.error(
                                                "Please provide your OpenRouter API key in the settings")
                                            break
                                        comparison_result = openrouter_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_openrouter_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            p_sampling=p_sampling,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "Ollama (Local)":
                                        comparison_result = ollama_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_ollama_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            p_sampling=p_sampling,
                                            max_tokens=max_tokens)
                                    else:  # OpenAI-compatible Server
                                        comparison_result = openai_compatible_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_openai_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)

                                    if comparison_result:
                                        feedback = comparison_result.get('detailed_feedback',
                                                                    'No feedback provided')
                                        score = score_calculator.extract_score(feedback)
                                        enhanced_feedback = score_calculator.generate_feedback(
                                            feedback, score)

                                        results.append({
                                            'filename': student_file.name,
                                            'student_id': student_id,
                                            'score': score,
                                            'feedback': enhanced_feedback
                                        })

                                        # Update DataFrame
                                        st.session_state.results_df = pd.DataFrame(results)

                                except Exception as e:
                                    st.error(
                                        f"Error processing {student_file.name}: {str(e)}")

                            # Update progress and timing
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            time_text.write(
                                f"Total processing time: {elapsed_time:.2f} seconds")

                        # Store results in session state
                        st.session_state.results = results

                        # Show completion message
                        st.success(
                            f"✅ Evaluation completed for {len(results)} submissions")

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

    # Results tab
    with results_tab:
        st.markdown("### All Scores")
        if st.session_state.results_df is None or st.session_state.results_df.empty:
            st.info("No results available yet. Please process some submissions first.")
        else:
            # Extract individual question scores from feedback
            def extract_question_scores(feedback):
                scores = {}
                for i in range(1, 4):  # Assuming 3 questions
                    match = re.search(f'Question {i}:\s*\[Score:\s*(\d+(?:\.\d+)?)/\d+\]', feedback)
                    scores[f'Q{i}'] = float(match.group(1)) if match else 0.0
                return pd.Series(scores)

            # Add question scores to the DataFrame
            question_scores = pd.DataFrame(st.session_state.results_df['feedback'].apply(extract_question_scores))
            scores_df = pd.concat([
                st.session_state.results_df[['student_id', 'filename', 'score']],
                question_scores
            ], axis=1).sort_values('score', ascending=False)

            # Display the enhanced scores table
            st.dataframe(
                scores_df,
                column_config={
                    'student_id': 'Student ID',
                    'filename': 'Filename',
                    'score': st.column_config.NumberColumn('Total Score (/20)', format="%.1f"),
                    'Q1': st.column_config.NumberColumn('Question 1 (/8)', format="%.1f"),
                    'Q2': st.column_config.NumberColumn('Question 2 (/4)', format="%.1f"),
                    'Q3': st.column_config.NumberColumn('Question 3 (/8)', format="%.1f")
                },
                hide_index=True
            )

            st.markdown("### Detailed Results")
            for result in st.session_state.results:
                with st.expander(f"📄 {result['filename']} (Student ID: {result['student_id']})"):
                    display_feedback(result['feedback'], result['score'],
                                  f"result_{result['student_id']}")

    # Statistical Analysis tab
    with stats_tab:
        st.markdown("### Statistical Analysis")
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results)

            # Update statistical visualizations
            st.session_state.stats_data = {
                'boxplot': statistical_analyzer.generate_score_boxplot(df['score'].tolist()),
                'distribution': statistical_analyzer.generate_score_distribution(df['score'].tolist())['plot'],
                'performance': statistical_analyzer.generate_performance_pie_chart(df['score'].tolist()),
                'bands': statistical_analyzer.generate_performance_bands(df['score'].tolist())
            }

            # Create tabs for different visualizations
            dist_tab, box_tab, quest_tab, perf_tab = st.tabs(
                ['Score Distribution', 'Box Plot', 'Question Analysis', 'Performance Bands']
            )

            # Score distribution tab
            with dist_tab:
                st.plotly_chart(st.session_state.stats_data['distribution'], use_container_width=True)

            # Box plot tab
            with box_tab:
                st.plotly_chart(st.session_state.stats_data['boxplot'], use_container_width=True)

            # Question analysis tab
            with quest_tab:
                # Generate and display the question scores chart
                question_scores_chart = statistical_analyzer.generate_question_scores_chart(st.session_state.results)
                st.plotly_chart(question_scores_chart, use_container_width=True)

                # Add explanation of the visualization
                st.markdown("""
                **Chart Explanation:**
                - Each box shows the distribution of scores for a specific question
                - Individual points represent student scores
                - The dotted line shows the mean score
                - Percentage shows the average achievement rate for each question
                """)

            # Performance bands tab
            with perf_tab:
                st.plotly_chart(st.session_state.stats_data['performance'], use_container_width=True)

                # Display performance bands table
                if st.session_state.stats_data['bands']:
                    st.markdown("### Performance Bands Breakdown")
                    bands_df = pd.DataFrame(
                        list(st.session_state.stats_data['bands'].items()),
                        columns=['Band', 'Count']
                    )
                    st.dataframe(bands_df, use_container_width=True)

    # Reports tab
    with reports_tab:
        st.header("📊 Individual Reports")

        # Add download button for CSV export at the top of Reports tab
        if st.session_state.results:
            col1, col2 = st.columns([1, 3])
            with col1:
                results_df = pd.DataFrame(st.session_state.results)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results as CSV",
                    data=csv,
                    file_name=f"assignment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download all grading results in CSV format"
                )
            st.markdown("---")  # Add a separator line

        if st.session_state.results:
            st.markdown("### Generate Reports")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Generate Complete Report", use_container_width=True):
                    with st.spinner("Generating complete report..."):
                        try:
                            report_pdf = report_generator.generate_complete_report(
                                st.session_state.results,
                                st.session_state.results_df,
                                marking_criteria,
                                statistical_analyzer
                            )
                            st.download_button(
                                "📥 Download Complete Report",
                                report_pdf,
                                "complete_report.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating complete report: {str(e)}")

            with col2:
                if st.button("Download All Individual Reports", use_container_width=True):
                    with st.spinner("Generating individual reports..."):
                        try:
                            # Create a ZIP file in memory
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for result in st.session_state.results:
                                    # Generate individual PDF report
                                    pdf_data = report_generator.generate_pdf_report(result)

                                    # Add PDF to ZIP with a meaningful filename
                                    filename = f"report_{result['student_id']}_{result['filename'].split('.')[0]}.pdf"
                                    zip_file.writestr(filename, pdf_data)

                            # Prepare ZIP file for download
                            zip_buffer.seek(0)
                            st.download_button(
                                "📥 Download All Individual Reports (ZIP)",
                                zip_buffer.getvalue(),
                                "individual_reports.zip",
                                "application/zip",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating individual reports: {str(e)}")

            # Individual report generation section
            st.markdown("### Individual Reports")
            for result in st.session_state.results:
                with st.expander(f"📄 Generate report for {result['filename']} (Student ID: {result['student_id']})"):
                    try:
                        report_pdf = report_generator.generate_pdf_report(result)
                        st.download_button(
                            "📥 Download Individual Report",
                            report_pdf,
                            f"report_{result['student_id']}_{result['filename'].split('.')[0]}.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating individual report: {str(e)}")
            # Add creator credits at the bottom of the page
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666666; padding: 20px;'>
        <p>Created by the University of Alberta, Department of Civil & Environmental Engineering. By: Mohamed sabek</p>
        <p> 2024 University of Alberta. All rights reserved.</p>
        </div>
        """,
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()