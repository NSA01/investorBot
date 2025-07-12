#!/usr/bin/env python3
"""
Helper script to create .env file with API keys
"""

import os

def create_env_file():
    """Create .env file with template"""
    
    env_content = """# API Keys - Replace with your actual keys
TAVILY_API_KEY=your-tavily-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your-secret-key-here

# File Configuration
EXCEL_FILE=linkedin_user_posts_1752243703.xlsx
"""
    
    if os.path.exists('.env'):
        print("⚠️ .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled.")
            return
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("📝 Please edit the .env file and add your actual API keys:")
        print("   - TAVILY_API_KEY: Get from https://tavily.com/")
        print("   - OPENAI_API_KEY: Get from https://platform.openai.com/")
        print("")
        print("🔒 The .env file is already in .gitignore to keep your keys secure.")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    print("🔧 Creating .env file for Insight AI...")
    create_env_file() 