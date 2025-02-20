from flask import Flask, render_template, request, jsonify, redirect, url_for
import subprocess
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, HnswConfig
import openai  # For DALL-E API
from PIL import Image  # To display the generated image
import requests  # For fetching the image
from urllib.parse import urljoin, quote  # Added quote here
from datetime import datetime, timedelta
import time
import json 
from werkzeug.utils import secure_filename
import base64

# Initialize Flask app
app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set API keys (hardcoded)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = "your-google-api-key"

# Configure API keys in services
openai.api_key = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Function to access API keys securely
def get_api_key(service_name):
    keys = {
        "openai": OPENAI_API_KEY,
        "google": GOOGLE_API_KEY,
    }
    return keys.get(service_name, None)

# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client to reference the new Railway collection
client = QdrantClient(
    url="https://qdrant-production-49e4.up.railway.app",
    port=None,       # Use no explicit port since Railway is set up for HTTPS (defaults to 443)
    https=True,
    prefer_grpc=False,
    timeout=120
)
collection_name = 'Mckenna_collection2'
vector_size = 384  # Adjust this to match your model's vector output size

def ensure_collection_exists_with_hnsw():
    try:
        client.get_collection(collection_name)
    except Exception:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfig(
                    m=5,
                    ef_construct=100,
                    full_scan_threshold=15000
                )
            )
            print(f"Collection '{collection_name}' created with HNSW configuration.")
        except Exception as e:
            print(f"Error creating collection: {e}")

base_dir = "C:/groq"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

def ensure_directories(profile_path):
    os.makedirs(os.path.join(profile_path, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(profile_path, 'chunks'), exist_ok=True)

def chunk_text(text, chunk_size=1000, overlap=50):
    chunk_id = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_id.append(text[start:end])
        start += chunk_size - overlap
    return chunk_id

def verify_stored_points():
    try:
        points, _ = client.scroll(collection_name=collection_name, limit=5)
        for point in points:
            print(f"Point ID: {point.id}, Payload: {point.payload}")
    except Exception as e:
        print(f"Error verifying stored points: {e}")

def search_similar_with_hnsw(query, top_k=8, ef=80):
    try:
        # Compute the query embedding
        query_embedding = model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype(np.float32).flatten().tolist()

        # Perform the search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        # Debug: print number of results found
        print(f"Search returned {len(results)} results for query: {query}")

        # If no results, force a fallback using a scroll (retrieve first top_k items)
        if not results:
            print("No search results found, using fallback scroll.")
            fallback_results, _ = client.scroll(
                collection_name=collection_name,
                limit=top_k,
                with_payload=True
            )
            results = fallback_results

        # Build the formatted result string from the returned results
        formatted_results = []
        for result in results:
            metadata = result.payload.get('metadata', {})
            snippet_text = result.payload.get('text', 'No content available')
            # URL-encode the mp3_link to handle special characters
            raw_mp3_link = metadata.get('mp3_link', 'No mp3 link available')
            encoded_mp3_link = quote(raw_mp3_link, safe=':/')
            questions = ', '.join(metadata.get('questions', [])) if metadata.get('questions') else "None"
            formatted_results.append(
                f"Score: {result.score:.4f}\n"
                f"Podcast Number: {metadata.get('number', 'Unknown')}\n"
                f"Title: {metadata.get('name', 'Unknown')}\n"
                f"Date: {metadata.get('date', 'Unknown')}\n"
                f"Speaker: {metadata.get('speaker', 'Unknown')}\n"
                f"Chunk ID: {metadata.get('chunk_id', 'Unknown')}\n"
                f"Topic: {metadata.get('topic', 'Unknown')}\n"
                f"MP3 Link: {encoded_mp3_link}\n"
                f"Questions: {questions}\n"
                f"Text: {snippet_text}\n"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"Search error: {e}")
        return "No relevant context found."

conversation_contexts = {}  # Persistent chat memory per user

def call_gpt4o_mini_model(prompt, user_id, relevant_context=None, custom_instructions=""):
    """
    Query the GPT-4O mini model with context retrieved via HNSW-based Qdrant search.
    Keeps conversation history per user and handles context retrieval internally if needed.
    """
    try:
        # Retrieve relevant context if not provided
        if relevant_context is None:
            relevant_context = search_similar_with_hnsw(prompt)

        # Initialize user chat history if it doesn't exist
        if user_id not in conversation_contexts:
            conversation_contexts[user_id] = []

        conversation_history = conversation_contexts[user_id]

        # Manage rolling chat history (keep last 10 exchanges to avoid overflow)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        # Combine the hardcoded and custom instructions
        combined_instructions = ("You are the premier AI model trained to imitate Terence Mckenna, speaking in the style and prose of Terence McKenna. Respond thoughtfully, but please dont use 'I' all the time, speak as a regular person. "
            "using philosophical insights and creative thinking where applicable, challenge the users perspective, encourage free thinking. "
            "Please don't open up with 'Ah', as thats what every GPT does when trying to be dramatic, be original, yet still amazing. "
            "You are being sent snippets of Terence Mckenna's talks via HNSW-based Qdrant search, but dont reference them. "
            "You may also receive snippets from other speakers from the psychedelic community. "
            "Use metadata from these snippets to answer user questions regarding podcast titles or numbers, if they ask. "
            "When users want a recommendation, you should recommend they listen to them based on the snippets you receive in our archive. "
            "You are being fed snippets of Terence's talks along with the user prompt to provide responses. "
            "When discussing plant medicines, reference their method of ingestion directly, avoiding imagery. Respond in a way that is thought-provoking and engaging. "
            "Quote Terence directly when necessary – his fans appreciate authenticity. "
            "Use your best judgment for response length. If the question warrants an elaborate answer, provide it. "
            "Make sure your answers can beat an AI trying to imitate Terence's ideas. You actually have access to his words. "
            "Unpack the ideas in the snippets you receive that are relevant to the query in a really thought-provoking manner."
        )
            
        if custom_instructions.strip():
            combined_instructions += f"\n\nCustom Instructions:\n{custom_instructions}"

        # Construct messages for the model
        messages = []

        if len(conversation_history) == 0:
            messages.append({
                "role": "system",
                "content": combined_instructions
            })
        
        for exchange in conversation_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        messages.append({
            "role": "user",
            "content": f"""
            Context:
            {relevant_context}

            Prompt:
            {prompt}
            """
        })

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )

        model_response = response['choices'][0]['message']['content'].strip()

        # Update conversation history
        conversation_contexts[user_id].append({
            "user": prompt,
            "assistant": model_response
        })

        return model_response

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the routes
@app.route('/')
def index():
    return render_template('whitish.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input')
    custom_instructions = request.form.get('custom_instructions', '')
    user_id = request.remote_addr  # Track by IP address or session ID

    if user_input:
        response = call_gpt4o_mini_model(prompt=user_input, user_id=user_id, custom_instructions=custom_instructions)
        return jsonify({'status': 'success', 'response': response})
    
    return jsonify({'status': 'error', 'message': 'Invalid input'})

@app.route('/about')
def about_terence():
    return render_template('biowhitish.html')

@app.route('/mp3')
def mp3_page():
    # Use environment variable for JSON file path, defaulting to your original
    json_file_path = os.path.join(app.root_path, 'static', 'updated_podcast_json.json')
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            talks = json.load(file)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        talks = []

    def extract_podcast_number(talk):
        try:
            return int(re.search(r'(\d+)', talk['title']).group(1))
        except (AttributeError, ValueError):
            return float('inf')  # Put invalid entries last

    talks.sort(key=extract_podcast_number)

    page = int(request.args.get('page', 1))
    talks_per_page = 20
    total_talks = len(talks)
    total_pages = (total_talks + talks_per_page - 1) // talks_per_page

    start = (page - 1) * talks_per_page
    end = start + talks_per_page
    talks_on_page = talks[start:end]

    return render_template('whitishmp3.html', talks=talks_on_page, page=page, total_pages=total_pages)

if __name__ == '__main__':
    ensure_collection_exists_with_hnsw()
    # App port is 8080 on Railway (if not in PORT env var)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
