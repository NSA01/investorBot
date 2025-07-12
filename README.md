# Insight AI - Excel Analysis Dashboard

A powerful Flask-based web application that combines AI-powered chat functionality with Excel data analysis capabilities. The application features web search integration, streaming responses, and comprehensive data visualization.

## 🚀 Features

### Chat Functionality
- **AI-Powered Conversations**: GPT-4 integration for intelligent responses
- **Web Search Integration**: Real-time web search using Tavily API
- **Streaming Responses**: Real-time streaming of AI responses
- **Topic Guardians**: Intelligent topic filtering for relevant conversations
- **Conversation History**: Persistent chat history with session management

### Excel Analysis Dashboard
- **File Upload**: Drag & drop Excel/CSV file upload
- **Data Analysis**: Automatic data type detection and statistics
- **Interactive Charts**: Dynamic data visualization with Chart.js
- **Data Quality Assessment**: Missing values, duplicates, and quality scoring
- **Column Selection**: Interactive column selection for analysis
- **Data Preview**: Real-time data table preview

### Production Features
- **WSGI Deployment**: Gunicorn-based production deployment
- **Docker Support**: Containerized deployment with Docker
- **Nginx Integration**: Reverse proxy with load balancing
- **Security Headers**: Comprehensive security configuration
- **Rate Limiting**: API rate limiting and protection
- **Health Checks**: Application health monitoring

## 📋 Prerequisites

- Python 3.9+
- pip3
- Git
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Method 1: Direct Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insight-ai
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-lightweight.txt
   ```

4. **Set up environment variables**
   ```bash
   # Option A: Use the helper script
   python create_env.py
   
   # Option B: Create manually
   # Create a .env file with your API keys (see SETUP.md for details)
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

### Method 2: Using Docker

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build Docker image manually**
   ```bash
   docker build -t insight-ai .
   docker run -p 8000:8000 insight-ai
   ```

### Method 3: Production Deployment

1. **Use the deployment script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Or deploy manually with Gunicorn**
   ```bash
   gunicorn --config gunicorn.conf.py wsgi:app
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
TAVILY_API_KEY=your-tavily-api-key
OPENAI_API_KEY=your-openai-api-key

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Optional: Database and Redis
DATABASE_URL=postgresql://user:password@localhost/dbname
REDIS_URL=redis://localhost:6379/0
```

### Gunicorn Configuration

The `gunicorn.conf.py` file contains optimized settings for production:

- **Workers**: Automatically calculated based on CPU cores
- **Timeout**: 30 seconds for request handling
- **Logging**: Comprehensive access and error logging
- **Security**: Process isolation and security headers

### Nginx Configuration

The `nginx.conf` file provides:

- **Reverse Proxy**: Routes requests to Gunicorn
- **Load Balancing**: Support for multiple application instances
- **Rate Limiting**: API protection against abuse
- **SSL Support**: HTTPS configuration (uncomment to enable)
- **Gzip Compression**: Optimized content delivery

## 📊 Usage

### Chat Interface

1. Navigate to `http://localhost:8000`
2. Ask questions about investments, fintech, startups, or business topics
3. The AI will automatically use web search when needed
4. View conversation history and sources

### Excel Analysis Dashboard

1. Navigate to `http://localhost:8000/dashboard`
2. Upload an Excel or CSV file
3. Select columns for analysis
4. View statistics, charts, and data quality reports
5. Export analysis results

## 🚀 Deployment Options

### 1. Systemd Service (Linux)

The deployment script automatically creates a systemd service:

```bash
# Start the service
sudo systemctl start insight-ai

# Enable auto-start
sudo systemctl enable insight-ai

# View logs
sudo journalctl -u insight-ai -f
```

### 2. Docker Compose

For containerized deployment with Nginx:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Cloud Deployment

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn --config gunicorn.conf.py wsgi:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS/GCP/Azure
- Use the Docker image with container services
- Deploy with load balancers and auto-scaling
- Configure SSL certificates for HTTPS

## 🔒 Security Features

- **API Key Protection**: Environment variables for secure credential management
- **Rate Limiting**: API protection against abuse
- **Security Headers**: XSS protection, content type validation
- **File Upload Validation**: Secure file handling
- **Session Security**: Secure cookie configuration
- **Process Isolation**: Non-root user execution
- **Git Protection**: .gitignore prevents accidental commit of sensitive files

### ⚠️ Important Security Note

**API keys are no longer hardcoded in the source code.** The application now uses environment variables for secure credential management. This prevents accidental exposure of sensitive credentials in version control.

- Create a `.env` file with your API keys (see `SETUP.md` for detailed instructions)
- The `.env` file is automatically ignored by Git
- Never commit API keys or sensitive credentials to version control

## 📈 Monitoring

### Health Checks
- Application health endpoint: `/health`
- Docker health checks configured
- Systemd service monitoring

### Logging
- Access logs: Request/response logging
- Error logs: Application error tracking
- Gunicorn logs: Worker process monitoring

### Metrics
- Request count and response times
- Error rates and status codes
- Resource usage monitoring

## 🛠️ Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest
```

### Code Quality
```bash
# Install linting tools
pip install flake8 black

# Format code
black .

# Check code quality
flake8 .
```

## 📝 API Documentation

### Chat Endpoints
- `POST /chat/stream` - Streaming chat responses
- `POST /save-chat` - Save conversation history
- `GET /history` - Retrieve chat history

### Excel Analysis Endpoints
- `POST /upload-excel` - Upload Excel files
- `POST /analyze-excel` - Analyze uploaded data
- `GET /dashboard` - Dashboard interface

### Utility Endpoints
- `GET /health` - Health check
- `GET /` - Main chat interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for error details

## 🔄 Updates

To update the application:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart the service
sudo systemctl restart insight-ai
```

---

**Note**: Remember to update your API keys and configuration before deploying to production! #   i n v e s t o r B o t 
 
 
