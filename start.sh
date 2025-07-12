#!/bin/bash

echo "🚀 Starting Insight AI on Render..."

# Check if gunicorn is installed
echo "📦 Checking installed packages..."
pip list | grep gunicorn

# Check Python version
echo "🐍 Python version:"
python --version

# Check environment variables
echo "🔑 Environment variables:"
echo "PORT: $PORT"
echo "FLASK_ENV: $FLASK_ENV"
echo "TAVILY_API_KEY: ${TAVILY_API_KEY:0:10}..."
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# Start the application
echo "🌐 Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --workers 2 