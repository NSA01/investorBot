from flask import Flask, render_template, request, jsonify, session, Response, stream_template
import pandas as pd
import numpy as np
from tavily import TavilyClient
import os
from datetime import datetime
import json
import re
from openai import OpenAI
import time
from werkzeug.utils import secure_filename

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Make sure to set environment variables manually.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Configuration
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
EXCEL_FILE = os.getenv('EXCEL_FILE', 'linkedin_user_posts_1752243703.xlsx')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate required environment variables
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Topic guardians - keywords to stay on topic
ALLOWED_TOPICS = [
    'investment', 'investments', 'investing', 'investor', 'investors',
    'fintech', 'financial technology', 'financial tech',
    'startup', 'startups', 'venture capital', 'vc',
    'funding', 'fund', 'funds', 'capital', 'money',
    'business', 'company', 'companies', 'market', 'markets',
    'technology', 'tech', 'innovation', 'digital',
    'banking', 'bank', 'financial', 'finance', 'financing',
    'crypto', 'cryptocurrency', 'blockchain', 'defi',
    'ai', 'artificial intelligence', 'machine learning', 'ml',
    'data', 'analytics', 'analysis', 'research'
]

class ChatBot:
    def __init__(self):
        self.posts_data = []
        self.load_data()
    
    def load_data(self):
        """Load LinkedIn posts from Excel (without embeddings for fast startup)"""
        try:
            df = pd.read_excel(EXCEL_FILE)
            self.posts_data = df.to_dict('records')
            print(f"Loaded {len(self.posts_data)} posts (embeddings will be created on-demand)")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.posts_data = []
    
    def check_topic_relevance_openai(self, text):
        """Use OpenAI to check if the text is relevant to allowed topics"""
        try:
            system_prompt = f"""You are a conversational topic guardian for a fintech and investment chatbot named Awn. 
            
            The user is asking: "{text}"
            
            Determine if this question should be answered by Awn. Consider:
            1. Is it a general question about Awn's identity, capabilities, or how to use the system? (ALLOW)
            2. Is it a greeting, introduction, or conversational opener? (ALLOW)
            3. Is it about investments, fintech, startups, business, technology, or related topics? (ALLOW)
            4. Is it about sports, entertainment, personal life, or completely unrelated topics? (BLOCK)
            
            Be conversational and inclusive. If there's any reasonable connection to business, finance, technology, or if it's a general question about Awn, say YES.
            
            Respond with ONLY 'YES' or 'NO'."""
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == 'YES'
        except Exception as e:
            print(f"Error checking topic relevance: {e}")
            # Fallback to keyword-based check - be more inclusive
            text_lower = text.lower()
            
            # Allow general questions and greetings
            general_questions = ['who', 'what', 'how', 'why', 'when', 'where', 'do you', 'can you', 'are you', 'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'thanks', 'thank you']
            for word in general_questions:
                if word in text_lower:
                    return True
            
            # Check for business/fintech topics
            for topic in ALLOWED_TOPICS:
                if topic.lower() in text_lower:
                    return True
            
            # If it's a short question (likely conversational), allow it
            if len(text.split()) <= 5:
                return True
                
            return False
    
    def search_similar_posts(self, query, k=3):
        """Search for similar posts using OpenAI embeddings (on-demand)"""
        if len(self.posts_data) == 0:
            return []
        
        try:
            # Create query embedding using OpenAI
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Create embeddings for all posts and find similar ones
            similarities = []
            for i, post in enumerate(self.posts_data):
                try:
                    post_text = f"{post.get('title', '')} {post.get('content', '')}"
                    post_response = openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=post_text
                    )
                    post_embedding = post_response.data[0].embedding
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, post_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(post_embedding))
                    similarities.append((similarity, post))
                except Exception as e:
                    print(f"Error creating embedding for post {i}: {e}")
                    continue
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [post for _, post in similarities[:k]]
            
        except Exception as e:
            print(f"Error searching similar posts: {e}")
            return []
    
    def search_web(self, query):
        """Search the web using Tavily API"""
        try:
            print(f"Tavily search query: {query}")
            
            # Validate API key
            if not TAVILY_API_KEY or TAVILY_API_KEY == 'your-tavily-api-key':
                print("Error: Invalid Tavily API key")
                return []
            
            # Do a general search
            response = tavily_client.search(
                query=query, 
                search_depth="advanced", 
                max_results=8,
                include_answer=True,
                include_raw_content=False
            )
            
            if isinstance(response, dict):
                results = response.get('results', [])
                print(f"Found {len(results)} search results")
                return results
            else:
                print(f"Unexpected response type: {type(response)}")
                return []
                
        except Exception as e:
            print(f"Error searching web: {e}")
            return []
    
    def generate_response_stream(self, user_message):
        """Generate streaming response using OpenAI"""
        try:
            # Check topic relevance first
            if not self.check_topic_relevance_openai(user_message):
                return {
                    'response': "Hello! I'm Awn, your AI assistant specializing in investments, fintech, startups, and business insights. I'm here to help you with questions about these topics, or you can ask me general questions about my capabilities. How can I assist you today?",
                    'source': 'topic_guardian',
                    'similar_posts': [],
                    'web_results': []
                }
            
            # Search similar posts for context (optional, can be slow)
            similar_posts = []
            try:
                similar_posts = self.search_similar_posts(user_message, k=2)
            except Exception as e:
                print(f"Error searching similar posts: {e}")
            
            # Create context from similar posts
            context = ""
            if similar_posts:
                context = "Based on LinkedIn posts:\n" + "\n\n".join([
                    f"Post: {post.get('title', '')}\n{post.get('content', '')[:300]}..."
                    for post in similar_posts
                ])
            
            # Create system prompt for OpenAI
            system_prompt = f"""You are Awn, an expert AI assistant specializing in investments, fintech, startups, and business insights.

Current date: {datetime.now().strftime('%Y-%m-%d')}

Available context from LinkedIn posts:
{context}

Instructions:
1. If the LinkedIn context is relevant and sufficient, use it to provide a detailed answer
2. If the LinkedIn context is not relevant or insufficient, use the web search tool to get current information
3. Always provide helpful, accurate, and professional responses
4. Focus on practical insights and actionable information
5. For time-sensitive questions or recent events, use web search
6. Be aware of the current date and don't make statements about future events as if they're past

CRITICAL GUIDELINES:
1. You are Awn, an expert AI assistant specializing in investments, fintech, startups, and business insights.
2. If the user asked you in Arabic, respond in Arabic 
3. If the user asked you in English, respond in English
4. Current date is {datetime.now().strftime('%Y-%m-%d')} - be accurate about time
5. For questions about recent events, future predictions, or time-sensitive information, use web search

Current question: {user_message}"""

            # Define the web search tool
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for current information about a topic",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find current information"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            # Generate response using OpenAI with function calling
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                tools=tools,
                tool_choice="auto",
                stream=True,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            # Return a simple error response instead of None
            return {
                'response': f"Hello! I'm Awn, your AI assistant specializing in investments, fintech, startups, and business insights. I'm here to help you with any questions about these topics. How can I assist you today?",
                'source': 'openai',
                'error': str(e)
            }

# Initialize chatbot
chatbot = ChatBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if not user_message.strip():
        return jsonify({'error': 'Message cannot be empty'})
    
    # Generate response
    response_stream = chatbot.generate_response_stream(user_message)
    
    if response_stream is None:
        return jsonify({'error': 'Failed to generate response'})
    
    # Store in session history
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # For now, return a simple response (streaming will be handled separately)
    return jsonify({
        'response': 'Streaming response will be handled by the frontend',
        'source': 'openai',
        'streaming': True
    })

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    user_message = request.json.get('message', '')
    
    if not user_message.strip():
        return jsonify({'error': 'Message cannot be empty'})
    
    def generate():
        try:
            response_stream = chatbot.generate_response_stream(user_message)
            
            # Handle case where response_stream is a dict (error case)
            if isinstance(response_stream, dict):
                if 'error' in response_stream:
                    # This is an error response, send it as a single message
                    yield f"data: {json.dumps({'content': response_stream['response'], 'source': response_stream['source']})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                else:
                    # This is a regular response (like topic guardian)
                    yield f"data: {json.dumps({'content': response_stream['response'], 'source': response_stream['source']})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
            
            if response_stream is None:
                yield f"data: {json.dumps({'error': 'Failed to generate response'})}\n\n"
                return
            
            full_response = ""
            source = "openai"
            
            # Collect all chunks and tool calls
            tool_calls = []
            current_tool_call = None
            
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content, 'source': source})}\n\n"
                
                # Handle tool calls - they can be split across multiple chunks
                if chunk.choices[0].delta.tool_calls:
                    for tool_call_delta in chunk.choices[0].delta.tool_calls:
                        # Initialize new tool call if needed
                        if tool_call_delta.index is not None:
                            if current_tool_call is None or current_tool_call.get('index') != tool_call_delta.index:
                                current_tool_call = {
                                    'index': tool_call_delta.index,
                                    'id': tool_call_delta.id,
                                    'type': tool_call_delta.type,
                                    'function': {'name': '', 'arguments': ''}
                                }
                                tool_calls.append(current_tool_call)
                        
                        # Update tool call with delta information
                        if current_tool_call is not None:
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    current_tool_call['function']['name'] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    current_tool_call['function']['arguments'] += tool_call_delta.function.arguments
            
            # Execute tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call['function']['name'] == "web_search":
                        try:
                            # Parse the function arguments
                            args = json.loads(tool_call['function']['arguments'])
                            search_query = args.get('query', user_message)
                            
                            print(f"Executing web search for: {search_query}")
                            
                            # Send "performing web search" message
                            yield f"data: {json.dumps({'content': '\n\n🔍 **Performing a web search...**\n', 'source': 'searching'})}\n\n"
                            
                            # Execute web search
                            web_results = chatbot.search_web(search_query)
                            
                            if web_results and len(web_results) > 0:
                                # Send sources first
                                sources_text = "\n\n📚 **Sources:**\n"
                                max_sources = min(5, len(web_results))  # Show up to 5 sources
                                for i, result in enumerate(web_results[:max_sources]):
                                    url = result.get('url', 'Unknown')
                                    title = result.get('title', 'No title')
                                    # Truncate long titles for better display
                                    display_title = title[:80] + "..." if len(title) > 80 else title
                                    sources_text += f"{i+1}. [{display_title}]({url})\n"
                                
                                if len(web_results) > max_sources:
                                    sources_text += f"\n*... and {len(web_results) - max_sources} more sources*"
                                
                                yield f"data: {json.dumps({'content': sources_text, 'source': 'sources'})}\n\n"
                                
                                # Create comprehensive context from web results
                                web_context = "Based on recent web search results:\n\n"
                                for i, result in enumerate(web_results[:3]):
                                    web_context += f"Source {i+1}: {result.get('title', 'No title')}\n"
                                    web_context += f"URL: {result.get('url', 'Unknown')}\n"
                                    web_context += f"Content: {result.get('content', '')[:400]}...\n\n"
                                
                                # Generate response based on web results
                                web_system_prompt = f"""You are Awn, an expert AI assistant specializing in investments, fintech, startups, and business insights.

Current date: {datetime.now().strftime('%Y-%m-%d')}

Use the following web search results to provide a comprehensive, well-structured answer to the user's question:

{web_context}

INSTRUCTIONS:
1. Provide a detailed, informative response based on the search results
2. Include specific facts, figures, and insights from the sources
3. Structure your response with clear sections if appropriate
4. Cite specific sources when mentioning facts or quotes
5. If there are multiple relevant sources, synthesize the information
6. Focus on practical insights and actionable information
7. Be conversational but professional
8. If the information is time-sensitive, mention the current date context

FORMAT:
- Start with a brief overview
- Provide detailed analysis with specific examples
- Include relevant statistics or data points
- End with key takeaways or implications

Stream your response naturally and engagingly."""

                                web_response = openai_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": web_system_prompt},
                                        {"role": "user", "content": user_message}
                                    ],
                                    stream=True,
                                    temperature=0.7
                                )
                                
                                # Stream the web search response
                                yield f"data: {json.dumps({'content': '\n\n💡 **Answer:**\n', 'source': 'web_search'})}\n\n"
                                
                                for chunk in web_response:
                                    if chunk.choices[0].delta.content:
                                        content = chunk.choices[0].delta.content
                                        yield f"data: {json.dumps({'content': content, 'source': 'web_search'})}\n\n"
                                
                            else:
                                # No web results found
                                yield f"data: {json.dumps({'content': '\n\n❌ No current information found for this topic. Please try rephrasing your question.', 'source': 'no_data'})}\n\n"
                                
                        except Exception as e:
                            print(f"Error executing web search: {e}")
                            yield f"data: {json.dumps({'content': '\n\n⚠️ Web search encountered an error. Please try rephrasing your question.', 'source': 'error'})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            print(f"Error in streaming: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/health')
def health_check():
    """Health check endpoint for production deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-fast'
    })

@app.route('/history')
def get_history():
    return jsonify(session.get('chat_history', []))

@app.route('/dashboard')
def dashboard():
    # Create individual dashboard data
    dashboard_data = {
        'total_posts': len(chatbot.posts_data),
        'topics': ['Investments', 'Fintech', 'Startups', 'Venture Capital'],
        'recent_activity': [
            {'date': '2024-01-15', 'posts': 5, 'engagement': 120},
            {'date': '2024-01-14', 'posts': 3, 'engagement': 85},
            {'date': '2024-01-13', 'posts': 7, 'engagement': 200},
        ],
        'user_stats': {
            'total_chats': len(session.get('chat_history', [])),
            'favorite_topics': ['Fintech', 'Investment Opportunities'],
            'last_active': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    }
    return render_template('dashboard.html', data=dashboard_data)

if __name__ == '__main__':
    print("🚀 Starting Insight AI (Fast Mode)...")
    print("⚡ No heavy models loaded at startup - embeddings created on-demand")
    print("🌐 Server will be available at: http://localhost:5000")
    app.run(debug=True, port=5000) 