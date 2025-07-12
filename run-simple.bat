@echo off
echo 🚀 Starting Insight AI (Simple Mode)...

REM Check if .env file exists and load it
if exist .env (
    echo ✅ Loading environment variables from .env file...
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set %%a=%%b
        )
    )
) else (
    echo ⚠️ .env file not found. Please create one with your API keys.
    echo 📝 Example .env file:
    echo TAVILY_API_KEY=your-tavily-api-key-here
    echo OPENAI_API_KEY=your-openai-api-key-here
    pause
    exit /b 1
)

REM Set Flask environment variables
set FLASK_ENV=production
set FLASK_APP=wsgi.py

REM Validate required environment variables
if "%TAVILY_API_KEY%"=="" (
    echo ❌ TAVILY_API_KEY not set in .env file
    pause
    exit /b 1
)
if "%OPENAI_API_KEY%"=="" (
    echo ❌ OPENAI_API_KEY not set in .env file
    pause
    exit /b 1
)

echo ✅ Environment variables loaded successfully
echo 🔑 TAVILY_API_KEY: %TAVILY_API_KEY:~0,10%...
echo 🔑 OPENAI_API_KEY: %OPENAI_API_KEY:~0,10%...

echo 📦 Installing lightweight dependencies...
pip install -r requirements-lightweight.txt

echo 🌐 Starting Insight AI server...
echo 📍 Application will be available at: http://localhost:8000
echo 📊 Dashboard available at: http://localhost:8000/dashboard
echo 💚 Health check available at: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python run-lightweight.py

pause 