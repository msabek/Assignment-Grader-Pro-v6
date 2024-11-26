# Assignment Grader Pro v6.0.1

Assignment Grader Pro is an advanced AI-powered grading assistant that helps educators and teaching assistants streamline their grading process. The application leverages multiple AI models to provide consistent, objective, and detailed feedback on student submissions.

## 🌟 Key Features

- **Multi-Model Support**: Compatible with OpenAI, Groq, Mistral, Ollama, and other AI providers
- **Batch Processing**: Grade multiple assignments simultaneously
- **Statistical Analysis**: Get insights into class performance and grading patterns
- **Customizable Criteria**: Define and adjust your marking scheme
- **Detailed Reports**: Generate comprehensive feedback and statistical reports
- **File Format Support**: Process PDF, TXT, and various text formats
- **Local Model Support**: Use Ollama for offline grading capabilities

## 📋 Prerequisites

- Windows 10 or later
- Python 3.8+
- Internet connection (for cloud AI models)
- API keys for chosen AI providers

## 🚀 Quick Start

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Create a `.env` file with your API keys
   - Or configure keys through the application interface

3. **Launch**
   ```bash
   # Option 1: Using the batch file
   run_grader.bat

   # Option 2: Direct launch
   streamlit run main.py
   ```

## 📚 Documentation

- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Quick Start Guide](QUICK_START.md) - Get started in minutes

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI Integration**: OpenAI, Groq, Mistral, Ollama
- **File Processing**: PyPDF2, python-docx
- **Analysis**: pandas, numpy

## 🔒 Security

- API keys are stored securely in environment variables
- Local processing options available for sensitive data
- No student data is retained after processing

## 📊 Features in Detail

1. **AI Model Selection**
   - Choose from multiple AI providers
   - Configure model parameters
   - Use local models for offline grading

2. **Assignment Processing**
   - Batch upload support
   - Multiple file format handling
   - Automated text extraction

3. **Grading Features**
   - Customizable rubrics
   - Automated feedback generation
   - Manual override options

4. **Analysis Tools**
   - Score distribution visualization
   - Performance metrics
   - Trend analysis
   - Export capabilities


Created by : Mohamed sabek , PhD researcher , AI expert and software developer at the University of Alberta 
