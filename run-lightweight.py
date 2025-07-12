#!/usr/bin/env python3
"""
Lightweight version of Insight AI that avoids PyTorch memory issues
"""

import os
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Make sure to set environment variables manually.")

# Set Flask environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'wsgi.py'

# Validate required environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

print("🚀 Starting Insight AI (Lightweight Mode)...")
print(f"🔑 TAVILY_API_KEY: {os.environ['TAVILY_API_KEY'][:10]}...")
print(f"🔑 OPENAI_API_KEY: {os.environ['OPENAI_API_KEY'][:10]}...")

try:
    # Import Flask and basic dependencies
    from flask import Flask, render_template, request, jsonify, session, Response, stream_template
    import pandas as pd
    import numpy as np
    from tavily import TavilyClient
    from openai import OpenAI
    import time
    import json
    import re
    from datetime import datetime
    from werkzeug.utils import secure_filename
    
    print("✅ Basic imports successful")
    
    # Create a simplified Flask app without heavy ML dependencies
    app = Flask(__name__)
    app.secret_key = 'your-secret-key-here'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
    
    # Initialize clients
    tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    print("✅ Clients initialized")
    
    # Simple routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0-lightweight'
        })
    
    @app.route('/chat/stream', methods=['POST'])
    def chat_stream():
        try:
            data = request.get_json()
            user_message = data.get('message', '')
            
            def generate():
                # Simple response without heavy ML
                response = f"Hello! I'm running in lightweight mode. You said: {user_message}"
                yield f"data: {json.dumps({'content': response, 'done': True})}\n\n"
            
            return Response(generate(), mimetype='text/plain')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    print("✅ Routes configured")
    
    # Import waitress for production server
    import waitress
    print("✅ Waitress imported successfully")
    
    print("🌐 Starting Insight AI server (Lightweight Mode)...")
    print("📍 Application will be available at: http://localhost:8000")
    print("📊 Dashboard available at: http://localhost:8000/dashboard")
    print("💚 Health check available at: http://localhost:8000/health")
    print("Press Ctrl+C to stop the server")
    
    # Start the server
    waitress.serve(app, host='0.0.0.0', port=8000, threads=2)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try installing basic dependencies: pip install flask waitress openai tavily-python")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1) 