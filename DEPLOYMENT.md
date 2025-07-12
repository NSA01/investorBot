# Deployment Guide for Render

## 🚀 Quick Deploy to Render

### Method 1: Using render.yaml (Recommended)

1. **Push your code to GitHub** (if not already done)
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" → "Blueprint"
   - Select your repository
   - Render will automatically detect the `render.yaml` file

3. **Set Environment Variables**:
   - In your Render dashboard, go to your service
   - Navigate to "Environment" tab
   - Add these environment variables:
     - `TAVILY_API_KEY`: Your Tavily API key
     - `OPENAI_API_KEY`: Your OpenAI API key

4. **Deploy**:
   - Click "Create Blueprint Instance"
   - Render will automatically build and deploy your app

### Method 2: Manual Deployment

1. **Create New Web Service**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   - **Name**: `insight-ai` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

3. **Set Environment Variables**:
   - `TAVILY_API_KEY`: Your Tavily API key
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `FLASK_ENV`: `production`

4. **Deploy**:
   - Click "Create Web Service"

## 🔧 Configuration Files

### render.yaml
```yaml
services:
  - type: web
    name: insight-ai
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16
      - key: TAVILY_API_KEY
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: FLASK_ENV
        value: production
```

### requirements.txt
```
flask==2.3.3
pandas==2.1.1
numpy==1.24.3
tavily-python==0.2.8
openai==1.3.0
python-dotenv==1.0.0
Werkzeug==2.3.7
openpyxl==3.1.2
gunicorn==21.2.0
```

## 🌐 Access Your App

After deployment, your app will be available at:
- **URL**: `https://your-app-name.onrender.com`
- **Health Check**: `https://your-app-name.onrender.com/health`

## 🔍 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check that `requirements.txt` exists
   - Verify all dependencies are listed
   - Check Python version compatibility

2. **App Won't Start**:
   - Verify environment variables are set
   - Check logs in Render dashboard
   - Ensure `gunicorn app:app` command works

3. **Port Issues**:
   - Render automatically sets the `PORT` environment variable
   - App should bind to `0.0.0.0:PORT`

4. **API Key Errors**:
   - Double-check your API keys in environment variables
   - Ensure keys are valid and have sufficient credits

### Logs and Monitoring:
- View logs in Render dashboard
- Monitor resource usage
- Check health endpoint: `/health`

## 💰 Cost Optimization

### Free Tier Limits:
- **Build Time**: 500 minutes/month
- **Runtime**: 750 hours/month
- **Bandwidth**: 100 GB/month

### Tips:
- Use lightweight dependencies
- Optimize startup time (already done with OpenAI embeddings)
- Monitor usage in Render dashboard

## 🔒 Security

### Environment Variables:
- Never commit API keys to Git
- Use Render's environment variable system
- Rotate keys regularly

### Production Best Practices:
- Use HTTPS (automatic on Render)
- Set appropriate CORS headers
- Implement rate limiting if needed
- Monitor for suspicious activity

## 📈 Scaling

### Upgrade Plans:
- **Free**: 1 instance, 512 MB RAM
- **Starter**: $7/month, 1 instance, 1 GB RAM
- **Standard**: $25/month, multiple instances, auto-scaling

### Performance Tips:
- Use connection pooling for databases
- Implement caching where appropriate
- Monitor response times
- Optimize database queries 