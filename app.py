from flask import Flask, render_template, request, jsonify, session, Response, stream_template
import pandas as pd
import faiss
import numpy as np
from tavily import TavilyClient
import os
from datetime import datetime
import json
import re
from openai import OpenAI
import time
import pandas as pd
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
        self.faiss_index = None
        self.posts_data = []
        self.load_and_index_data()
    
    def load_and_index_data(self):
        """Load LinkedIn posts from Excel and create FAISS index"""
        try:
            df = pd.read_excel(EXCEL_FILE)
            self.posts_data = df.to_dict('records')
            
            # Create embeddings for all posts using OpenAI
            texts = [f"{post.get('title', '')} {post.get('content', '')}" for post in self.posts_data]
            
            # Use OpenAI embeddings instead of SentenceTransformer
            embeddings = []
            for text in texts:
                try:
                    response = openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
                except Exception as e:
                    print(f"Error creating embedding for text: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 1536)  # OpenAI ada-002 dimension
            
            if embeddings:
                embeddings_array = np.array(embeddings)
                # Create FAISS index
                dimension = embeddings_array.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings_array.astype('float32'))
                
                print(f"Loaded {len(self.posts_data)} posts and created FAISS index using OpenAI embeddings")
            else:
                print("No embeddings created, skipping FAISS index")
                self.faiss_index = None
                
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
        """Search for similar posts using FAISS with OpenAI embeddings"""
        if self.faiss_index is None or len(self.posts_data) == 0:
            return []
        
        try:
            # Create query embedding using OpenAI
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = np.array([response.data[0].embedding])
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            similar_posts = []
            for idx in indices[0]:
                if idx < len(self.posts_data):
                    similar_posts.append(self.posts_data[idx])
            
            return similar_posts
        except Exception as e:
            print(f"Error searching similar posts: {e}")
            return []
    
    def search_web(self, query):
        """Search the web using Tavily API"""
        try:
            print(f"Tavily search query: {query}")
            print(f"Tavily API key: {TAVILY_API_KEY[:10]}...")
            
            # Validate API key
            if not TAVILY_API_KEY or TAVILY_API_KEY == 'your-tavily-api-key':
                print("Error: Invalid Tavily API key")
                return []
            
            # First, try to search specifically on fintechnews.ae for Saudi Arabia content
            fintechnews_query = f"site:fintechnews.ae Saudi Arabia fintech {query}"
            print(f"Searching fintechnews.ae specifically: {fintechnews_query}")
            
            try:
                fintechnews_response = tavily_client.search(
                    query=fintechnews_query,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False
                )
                
                fintechnews_results = []
                if isinstance(fintechnews_response, dict):
                    results = fintechnews_response.get('results', [])
                    for result in results:
                        if isinstance(result, dict) and result.get('content') and 'fintechnews.ae' in result.get('url', ''):
                            fintechnews_results.append(result)
                
                print(f"Found {len(fintechnews_results)} results from fintechnews.ae")
                
            except Exception as e:
                print(f"Error searching fintechnews.ae: {e}")
                fintechnews_results = []
            
            # Then do a general search
            general_response = tavily_client.search(
                query=query, 
                search_depth="advanced", 
                max_results=8,
                include_answer=True,
                include_raw_content=False
            )
            
            print(f"General search response type: {type(general_response)}")
            print(f"General search response keys: {general_response.keys() if isinstance(general_response, dict) else 'Not a dict'}")
            
            # Handle different response formats
            if isinstance(general_response, dict):
                general_results = general_response.get('results', [])
                print(f"Found {len(general_results)} general results")
                
                # Validate results
                if not general_results:
                    print("No results found in general search")
                    return fintechnews_results
                
                # Check if results have required fields
                valid_general_results = []
                for i, result in enumerate(general_results):
                    if isinstance(result, dict) and result.get('content'):
                        valid_general_results.append(result)
                    else:
                        print(f"Invalid result at index {i}: {result}")
                
                # Combine and prioritize results
                all_results = fintechnews_results + valid_general_results
                
                # Remove duplicates based on URL
                seen_urls = set()
                unique_results = []
                for result in all_results:
                    url = result.get('url', '')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)
                
                # Prioritize fintechnews.ae results
                prioritized_results = []
                fintechnews_found = False
                
                for result in unique_results:
                    url = result.get('url', '').lower()
                    title = result.get('title', '').lower()
                    content = result.get('content', '').lower()
                    
                    # Prioritize fintechnews.ae results first
                    if 'fintechnews.ae' in url:
                        if not fintechnews_found:
                            prioritized_results.insert(0, result)
                            fintechnews_found = True
                            print(f"Fintechnews.ae article prioritized: {result.get('title', 'No title')}")
                        continue
                    
                    # Then check for Saudi Arabia fintech content
                    saudi_keywords = ['saudi arabia', 'saudi', 'ksa', 'riyadh', 'jeddah', 'mecca', 'medina']
                    fintech_keywords = ['fintech', 'financial technology', 'digital banking', 'payment', 'blockchain', 'crypto', 'investment', 'startup', 'venture']
                    
                    has_saudi = any(keyword in title or keyword in content for keyword in saudi_keywords)
                    has_fintech = any(keyword in title or keyword in content for keyword in fintech_keywords)
                    
                    if has_saudi and has_fintech:
                        prioritized_results.append(result)
                        print(f"Saudi Arabia fintech article found: {result.get('title', 'No title')}")
                        continue
                    
                    prioritized_results.append(result)
                
                print(f"Final prioritized results: {len(prioritized_results)}")
                return prioritized_results
            else:
                print(f"Unexpected response type: {type(general_response)}")
                return fintechnews_results
                
        except Exception as e:
            print(f"Error searching web: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
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
            
            # Search similar posts for context
            similar_posts = self.search_similar_posts(user_message)
            
            # Create context from similar posts
            context = ""
            if similar_posts:
                context = "Based on LinkedIn posts:\n" + "\n\n".join([
                    f"Post: {post.get('title', '')}\n{post.get('content', '')[:300]}..."
                    for post in similar_posts[:2]
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
                            search_msg = json.dumps({'content': '\n\n🔍 **Performing a web search...**\n', 'source': 'searching'})
                            yield f"data: {search_msg}\n\n"
                            
                            # Send progress update for fintechnews.ae search
                            step1_msg = json.dumps({'content': '\n\n🔍 **Step 1:** Searching fintechnews.ae for Saudi Arabia fintech content...\n', 'source': 'progress'})
                            yield f"data: {step1_msg}\n\n"
                            
                            # Execute web search
                            web_results = chatbot.search_web(search_query)
                            
                            # Send progress update for general search
                            step2_msg = json.dumps({'content': '\n\n🌐 **Step 2:** Completed general search and prioritizing results...\n', 'source': 'progress'})
                            yield f"data: {step2_msg}\n\n"
                            
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
                                answer_msg = json.dumps({'content': '\n\n💡 **Answer:**\n', 'source': 'web_search'})
                                yield f"data: {answer_msg}\n\n"
                                
                                # Send progress update for response generation
                                step3_msg = json.dumps({'content': '\n\n📝 **Step 3:** Generating comprehensive response based on search results...\n', 'source': 'progress'})
                                yield f"data: {step3_msg}\n\n"
                                
                                for chunk in web_response:
                                    if chunk.choices[0].delta.content:
                                        content = chunk.choices[0].delta.content
                                        yield f"data: {json.dumps({'content': content, 'source': 'web_search'})}\n\n"
                                
                            else:
                                # No web results found
                                no_data_msg = json.dumps({'content': '\n\n❌ No current information found for this topic. Please try rephrasing your question.', 'source': 'no_data'})
                                yield f"data: {no_data_msg}\n\n"
                                
                        except Exception as e:
                            print(f"Error executing web search: {e}")
                            print(f"Tool call: {tool_call}")
                            import traceback
                            traceback.print_exc()
                            error_msg = json.dumps({'content': '\n\n⚠️ Web search encountered an error. Please try rephrasing your question.', 'source': 'error'})
                            yield f"data: {error_msg}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            print(f"Error in streaming: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/save-chat', methods=['POST'])
def save_chat():
    """Save chat to session after streaming is complete"""
    try:
        data = request.json
        user_message = data.get('user_message', '')
        bot_response = data.get('bot_response', '')
        source = data.get('source', 'openai')
        
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat(),
            'source': source
        })
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving chat: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/web-search', methods=['POST'])
def web_search():
    user_message = request.json.get('message', '')
    
    if not user_message.strip():
        return jsonify({'error': 'Message cannot be empty'})
    
    try:
        web_results = chatbot.search_web(user_message)
        
        # Generate response based on web results
        if web_results:
            context = "Based on recent web search results:\n" + "\n\n".join([
                f"Source: {result.get('url', 'Unknown')}\n{result.get('content', '')[:300]}..."
                for result in web_results[:2]
            ])
            
            system_prompt = f"""You are an expert AI assistant. Use the following web search results to answer the user's question:

{context}

Provide a comprehensive answer based on the search results. Include relevant insights and cite the sources when possible."""

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return jsonify({
                'response': answer,
                'source': 'web_search',
                'web_results': web_results
            })
        else:
            return jsonify({
                'response': "I couldn't find relevant information through web search. Please try rephrasing your question.",
                'source': 'no_data'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint for production deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
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

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    """Handle Excel file upload and return data for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file extension
        allowed_extensions = {'xlsx', 'xls', 'csv'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload Excel (.xlsx, .xls) or CSV files only.'})
        
        # Read the file
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})
        
        # Convert DataFrame to list of lists for JSON serialization
        data = df.values.tolist()
        columns = df.columns.tolist()
        
        # Store in session for later use
        session['uploaded_data'] = {
            'data': data,
            'columns': columns,
            'filename': secure_filename(file.filename)
        }
        
        return jsonify({
            'success': True,
            'data': {
                'data': data,
                'columns': columns,
                'filename': secure_filename(file.filename)
            },
            'columns': columns
        })
        
    except Exception as e:
        print(f"Error uploading Excel file: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze-excel', methods=['POST'])
def analyze_excel():
    """Analyze uploaded Excel data and return insights"""
    try:
        data = request.json
        selected_columns = data.get('selected_columns', [])
        
        if 'uploaded_data' not in session:
            return jsonify({'success': False, 'error': 'No data uploaded'})
        
        uploaded_data = session['uploaded_data']
        df_data = uploaded_data['data']
        columns = uploaded_data['columns']
        
        # Create DataFrame for analysis
        df = pd.DataFrame(df_data, columns=columns)
        
        # Basic statistics
        stats = {}
        for col_idx in selected_columns:
            if col_idx < len(columns):
                col_name = columns[col_idx]
                col_data = df.iloc[:, col_idx]
                
                # Check if column is numeric
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_data.isna().all():
                        stats[col_name] = {
                            'type': 'numeric',
                            'mean': float(numeric_data.mean()),
                            'median': float(numeric_data.median()),
                            'std': float(numeric_data.std()),
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'count': int(numeric_data.count())
                        }
                    else:
                        stats[col_name] = {
                            'type': 'text',
                            'unique_values': int(col_data.nunique()),
                            'most_common': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                            'count': int(col_data.count())
                        }
                except:
                    stats[col_name] = {
                        'type': 'text',
                        'unique_values': int(col_data.nunique()),
                        'most_common': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                        'count': int(col_data.count())
                    }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'total_rows': len(df_data),
            'total_columns': len(columns)
        })
        
    except Exception as e:
        print(f"Error analyzing Excel data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/summarize-excel', methods=['POST'])
def summarize_excel():
    """Generate AI-powered summary of uploaded Excel data"""
    try:
        if 'uploaded_data' not in session:
            return jsonify({'success': False, 'error': 'No data uploaded'})
        
        uploaded_data = session['uploaded_data']
        df_data = uploaded_data['data']
        columns = uploaded_data['columns']
        filename = uploaded_data['filename']
        
        # Create DataFrame for analysis
        df = pd.DataFrame(df_data, columns=columns)
        
        # Prepare data summary for OpenAI
        data_summary = f"""
        File: {filename}
        Total Rows: {len(df_data)}
        Total Columns: {len(columns)}
        Columns: {', '.join(columns)}
        
        Data Preview (first 5 rows):
        {df.head().to_string()}
        
        Data Types:
        {df.dtypes.to_string()}
        
        Missing Values:
        {df.isnull().sum().to_string()}
        """
        
        # Generate AI summary
        system_prompt = """You are an expert data analyst. Analyze the provided Excel data and create a comprehensive summary that includes:

1. **Data Overview**: Brief description of the dataset
2. **Key Insights**: Most important patterns, trends, or observations
3. **Data Quality**: Assessment of data completeness and quality
4. **Recommendations**: Suggestions for further analysis or data improvements
5. **Business Value**: Potential business implications or opportunities

Make the summary clear, actionable, and professional. Focus on insights that would be valuable for business decision-making."""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_summary}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_summary = response.choices[0].message.content
        
        return jsonify({
            'success': True,
            'summary': ai_summary,
            'filename': filename,
            'total_rows': len(df_data),
            'total_columns': len(columns)
        })
        
    except Exception as e:
        print(f"Error summarizing Excel data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat-excel', methods=['POST'])
def chat_excel():
    """Chat with the uploaded Excel data using AI"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Message cannot be empty'})
        
        if 'uploaded_data' not in session:
            return jsonify({'error': 'No Excel data uploaded. Please upload a file first.'})
        
        uploaded_data = session['uploaded_data']
        df_data = uploaded_data['data']
        columns = uploaded_data['columns']
        filename = uploaded_data['filename']
        
        # Create DataFrame
        df = pd.DataFrame(df_data, columns=columns)
        
        # Prepare context for AI
        data_context = f"""
        You are analyzing an Excel file named: {filename}
        
        Dataset Information:
        - Total Rows: {len(df_data)}
        - Total Columns: {len(columns)}
        - Column Names: {', '.join(columns)}
        
        Data Preview (first 10 rows):
        {df.head(10).to_string()}
        
        Data Types:
        {df.dtypes.to_string()}
        
        Basic Statistics:
        {df.describe().to_string() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'No numeric columns for statistics'}
        
        User Question: {user_message}
        
        Please answer the user's question based on this data. Be specific, provide insights, and reference the actual data when possible. If the question cannot be answered with the available data, explain what additional information would be needed.
        """
        
        system_prompt = """You are an expert data analyst and business intelligence assistant. Your role is to help users understand and analyze their Excel data.

Guidelines:
1. Answer questions based on the provided data
2. Provide specific insights and observations
3. Use numbers and statistics when relevant
4. Suggest additional analyses when appropriate
5. Be clear and professional in your responses
6. If a question cannot be answered with the available data, explain why and suggest what additional data might be needed

Always base your answers on the actual data provided, not on general knowledge."""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_context}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        ai_response = response.choices[0].message.content
        
        # Save to chat history
        if 'excel_chat_history' not in session:
            session['excel_chat_history'] = []
        
        session['excel_chat_history'].append({
            'user': user_message,
            'assistant': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'response': ai_response,
            'source': 'excel_data',
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error in Excel chat: {e}")
        return jsonify({'error': str(e)})

@app.route('/excel-chat-history')
def get_excel_chat_history():
    """Get chat history for Excel data conversations"""
    return jsonify(session.get('excel_chat_history', []))

if __name__ == '__main__':
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Starting Insight AI on port {port}...")
    print(f"🌐 Server will be available at: http://0.0.0.0:{port}")
    
    # Use 0.0.0.0 for production deployment
    app.run(host='0.0.0.0', port=port, debug=False) 