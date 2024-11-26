# Installation Guide for Assignment Grader Pro v3.2

## System Requirements

### Minimum Requirements
- Windows 10 or later
- Python 3.8 or later
- 4GB RAM
- 500MB free disk space
- Internet connection for cloud AI models

### Recommended Requirements
- Windows 10/11
- Python 3.10 or later
- 8GB RAM
- 1GB free disk space
- Stable internet connection

## Installation Steps

1. **Install Python**
   - Download Python from [python.org](https://www.python.org/downloads/)
   - During installation:
     - Check "Add Python to PATH"
     - Check "Install pip"
   - Verify installation by opening Command Prompt and typing:
     ```
     python --version
     pip --version
     ```

2. **Download Assignment Grader Pro**
   - Download the latest release
   - Extract the zip file to your preferred location
   - Avoid paths with special characters or spaces

3. **Set Up Virtual Environment (Recommended)**
   ```bash
   # Navigate to the application directory
   cd path/to/AssignmentGraderPro

   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   .venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   # Ensure pip is up to date
   python -m pip install --upgrade pip

   # Install requirements
   pip install -r requirements.txt
   ```

5. **Configure API Keys**
   - Create a `.env` file in the application directory
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_key_here
     GROQ_API_KEY=your_groq_key_here
     MISTRAL_API_KEY=your_mistral_key_here
     OPENROUTER_API_KEY=your_openrouter_key_here
     ```
   - (Optional) Configure Ollama for local models

6. **Launch the Application**
   - Option 1: Double-click `run_grader.bat`
   - Option 2: Run manually:
     ```bash
     streamlit run main.py
     ```

## Troubleshooting

### Common Issues

1. **Python not found**
   - Ensure Python is in PATH
   - Try running `where python` in Command Prompt
   - Reinstall Python with "Add to PATH" checked

2. **Dependencies Installation Fails**
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing dependencies one by one
   - Check for conflicting packages
   - Use `pip install --verbose` for detailed error messages

3. **Application Won't Start**
   - Check if all dependencies are installed
   - Verify Python version compatibility
   - Ensure Streamlit is installed correctly
   - Check for firewall blocking

4. **API Connection Issues**
   - Verify API keys in `.env` file
   - Check internet connection
   - Ensure no spaces in API keys
   - Try different AI providers

5. **File Processing Problems**
   - Install required system libraries
   - Check file permissions
   - Verify supported file formats

## Support

For additional help:
- Check our [GitHub Issues](https://github.com/AssignmentGraderPro/issues)
- Join our community Discord
- Contact support@assignmentgraderpro.com
