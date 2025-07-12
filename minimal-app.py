#!/usr/bin/env python3
"""
Minimal Flask app for testing deployment
"""

import os
from flask import Flask, jsonify
from datetime import datetime

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Validate required environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'message': 'Insight AI is running!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-minimal'
    })

@app.route('/test')
def test():
    return jsonify({
        'message': 'Test endpoint working!',
        'credentials': {
            'tavily_set': bool(os.environ.get('TAVILY_API_KEY')),
            'openai_set': bool(os.environ.get('OPENAI_API_KEY'))
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Minimal Insight AI...")
    print(f"🔑 TAVILY_API_KEY: {os.environ.get('TAVILY_API_KEY', 'NOT_SET')[:10]}...")
    print(f"🔑 OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', 'NOT_SET')[:10]}...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("💚 Health check at: http://localhost:8000/health")
    print("🧪 Test endpoint at: http://localhost:8000/test")
    
    try:
        from waitress import serve
        print("✅ Using Waitress server")
        serve(app, host='0.0.0.0', port=8000, threads=2)
    except ImportError:
        print("⚠️ Waitress not available, using Flask development server")
        app.run(host='0.0.0.0', port=8000, debug=False) 