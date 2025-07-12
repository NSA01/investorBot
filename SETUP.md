# Environment Setup Guide

## API Keys Required

This application requires the following API keys to function properly:

### 1. Create a `.env` file

Create a `.env` file in the root directory of this project with the following content:

```env
# API Keys - Replace with your actual keys
TAVILY_API_KEY=your-tavily-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your-secret-key-here

# File Configuration
EXCEL_FILE=linkedin_user_posts_1752243703.xlsx
```

### 2. Get Your API Keys

#### Tavily API Key
1. Go to [Tavily AI](https://tavily.com/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Replace `your-tavily-api-key-here` with your actual key

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Go to API Keys section
4. Create a new API key
5. Replace `your-openai-api-key-here` with your actual key

### 3. Install Dependencies

```bash
pip install python-dotenv
```

### 4. Security Notes

- **Never commit your `.env` file to version control**
- The `.env` file is already in `.gitignore` to prevent accidental commits
- Keep your API keys secure and don't share them publicly
- Consider using environment variables in production deployments

### 5. Running the Application

After setting up your `.env` file, you can run the application using any of these methods:

#### Option 1: Main Application
```bash
python app.py
```

#### Option 2: Lightweight Version
```bash
python run-lightweight.py
```

#### Option 3: Windows Batch File
```bash
run-simple.bat
```

### 6. Troubleshooting

If you get errors about missing API keys:
1. Make sure your `.env` file exists in the project root
2. Verify that your API keys are correctly set
3. Check that you have installed `python-dotenv`
4. Restart your terminal/command prompt after creating the `.env` file

### 7. Production Deployment

For production deployments, set environment variables directly on your server/hosting platform rather than using a `.env` file. 