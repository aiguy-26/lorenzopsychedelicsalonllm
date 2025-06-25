from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
import subprocess, re, os, numpy as np, json, uuid, time, base64
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfig
import openai
from PIL import Image
import requests
from urllib.parse import urljoin, quote
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import psycopg2
import boto3
from urllib.parse import quote
import tempfile
import soundfile as sf
from io import BytesIO
import os
from PIL import Image
from openai import OpenAIError

# public GCS bucket (same default you used elsewhere)
GCS_BUCKET = os.getenv("BUCKET_NAME", "psychedeli_salon_mp3s")


# Load PostgreSQL connection URL
DATABASE_URL = 'postgresql://postgres:PJeqquglCYpSoKtcJUrIDASdyhlUvqwD@yamabiko.proxy.rlwy.net:23023/railway'

# Initialize Flask
app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set your OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


PROMPT_PATH = os.path.join(BASE_DIR, "static", "system_prompts.json")
if not os.path.exists(PROMPT_PATH):
    raise FileNotFoundError(f"Could not find system_prompt.json at {PROMPT_PATH!r}")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPTS = json.load(f)


# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
client = QdrantClient(
    url="https://qdrant-production-49e4.up.railway.app",
    port=None,
    https=True,
    prefer_grpc=False,
    timeout=120
)

collection_name = 'Psychedelicsalon_collection'
vector_size = 384  # Adjust to match model's vector output size

def ensure_collection_exists_with_hnsw():
    """Ensure our Qdrant collection is created with HNSW config."""
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

def chunk_text(text, chunk_size=1000, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def search_similar_with_hnsw(query, top_k=8, ef=80):
    """Perform a semantic search in Qdrant to fetch relevant context."""
    try:
        query_embedding = model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype(np.float32).flatten().tolist()

        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        if not results:
            # fallback
            fallback_results, _ = client.scroll(
                collection_name=collection_name,
                limit=top_k,
                with_payload=True
            )
            results = fallback_results

        formatted_results = []
        for result in results:
            metadata = result.payload.get('metadata', {})
            snippet_text = result.payload.get('text', 'No content available')
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

# In-memory conversation cache
conversation_contexts = {}

def call_gpt4o_mini_model(
    prompt,
    user_id,
    chat_id=None,
    relevant_context=None,
    prompt_type="default",
    custom_instructions="",
    voice_mode=False
):
    """Calls GPT, returns the model's text response along with chat_id."""
    try:
        # 1️⃣ HNSW lookup for context
        if relevant_context is None:
            relevant_context = search_similar_with_hnsw(prompt)

        # 2️⃣ Ensure chat_id
        if chat_id is None:
            chat_id = str(uuid.uuid4())

        # 3️⃣ Init per-user history
        history = conversation_contexts.setdefault(user_id, [])
        if len(history) > 10:
            history[:] = history[-10:]

        # 4️⃣ Pick the base system prompt
        if prompt_type == "custom" and custom_instructions.strip():
            base = custom_instructions
        else:
            base = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])

        # 5️⃣ Layer on voice instructions if toggled
        voice_txt = SYSTEM_PROMPTS.get("voice", "")

        if voice_mode and voice_txt:
            base = f"{voice_txt.strip()}\n\n{base.strip()}"

        # 6️⃣ Build the message stack
        messages = [{"role": "system", "content": base}]
        for ex in history:
            messages.append({"role": "user",      "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({
            "role": "user",
            "content": f"Context:\n{relevant_context}\n\nPrompt:\n{prompt}"
        })

        # 7️⃣ Call the API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8
        )
        reply = response.choices[0].message.content.strip()

        # 8️⃣ Save to history
        history.append({"user": prompt, "assistant": reply})
        conversation_contexts[user_id] = history

        return reply, chat_id

    except OpenAIError as api_err:
        return f"❌ OpenAI API error: {api_err}", chat_id
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}", chat_id


# -------------------------
#   New: Theme Selection
# -------------------------
@app.route('/set_theme', methods=['POST'])
def set_theme():
    theme = request.form.get('theme', 'whitish')
    resp = make_response(redirect(request.referrer or url_for('index')))
    resp.set_cookie('theme', theme)
    return resp

def get_template(template_base):
    # Default theme is "classic"
    theme = request.cookies.get('theme', 'classic')

    if template_base == "index":
        if theme == "classic":
            return "classic.html"
        elif theme == "psychedelic":
            return "psychedelic.html"
        elif theme == "dark":
            return "dark.html"
        elif theme == "light":
            return "light.html"
        else:
            return "classic.html"

    elif template_base == "resource":
        # Always serve resource.html, ignoring theme
        return "resource.html"

    elif template_base == "mp3":
        if theme == "classic":
            return "classicmp3.html"
        elif theme == "psychedelic":
            return "psychedelicmp3.html"
        elif theme == "dark":
            return "darkmp3.html"
        elif theme == "light":
            return "lightmp3.html"
        else:
            return "classicmp3.html"

    else:
        # fallback for any other route
        return f"{template_base}.html"
    
#  transcribe_audio

from faster_whisper import WhisperModel

def transcribe_audio(file_path):
    model = WhisperModel("tiny", compute_type="int8", device="cpu")

    segments, _ = model.transcribe(file_path, vad_filter=True)

    result = []
    for segment in segments:
        result.append(segment.text)
    return " ".join(result)

# -------------------------
#   Chat History and API Routes
# -------------------------
@app.route("/get_chat", methods=["GET"])
def get_chat():
    chat_id = request.args.get("chat_id")
    if not chat_id:
        return jsonify({"error": "Must provide chat_id"}), 400
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, created_at
            FROM chat_messages
            WHERE chat_id = %s
            ORDER BY id ASC
        """, (chat_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        messages = []
        for r in rows:
            messages.append({
                'role': r[0],
                'content': r[1],
                'created_at': r[2].isoformat() if r[2] else None
            })
        return jsonify({"messages": messages})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/transcribe_audio", methods=["POST"])
def transcribe_audio_endpoint():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files['audio']
        temp_path = os.path.join(tempfile.gettempdir(), "uploaded_audio.webm")
        audio_file.save(temp_path)

        transcription = transcribe_audio(temp_path)

        return jsonify({"transcript": transcription})

    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": str(e)}), 500
("After transcription request")

import traceback

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        # 1️⃣ Parse incoming JSON payload
        data = request.get_json()
        print("Received payload:", data)

        user_input          = data.get('user_input', '').strip()
        prompt_type         = data.get('prompt_type', 'default')
        custom_instructions = data.get('custom_instructions', '').strip()
        voice_mode          = data.get('voice_mode', False)
        chat_id             = data.get('chat_id')
        user_id             = data.get('user_id')

        # 2️⃣ Validate required fields
        if not user_id:
            return jsonify({'status': 'error', 'message': 'user_id is required'}), 400
        if not user_input:
            return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

        # 3️⃣ Create new chat session if needed
        if not chat_id:
            chat_id = str(uuid.uuid4())
            title = user_input[:50]
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO chat_sessions (chat_id, user_id, title)
                           VALUES (%s, %s, %s)""",
                        (chat_id, user_id, title)
                    )

        # 4️⃣ Call our GPT wrapper, passing through persona + voice flags
        ai_response, updated_chat_id = call_gpt4o_mini_model(
            prompt              = user_input,
            user_id             = user_id,
            chat_id             = chat_id,
            prompt_type         = prompt_type,
            custom_instructions = custom_instructions,
            voice_mode          = voice_mode
        )
        chat_id = updated_chat_id or chat_id

        # 5️⃣ Persist both user and assistant messages
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_messages (chat_id, role, content)
                       VALUES (%s, %s, %s)""",
                    (chat_id, 'user', user_input)
                )
                cur.execute(
                    """INSERT INTO chat_messages (chat_id, role, content)
                       VALUES (%s, %s, %s)""",
                    (chat_id, 'assistant', ai_response)
                )

        # 6️⃣ Return the AI’s reply and the (possibly new) chat_id
        return jsonify({
            'status': 'success',
            'response': ai_response,
            'chat_id': chat_id
        })

    except Exception as e:
        print("Error in get_response:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

# -------------------------
#   Other Existing Routes
# -------------------------
@app.route('/')
def index():
    return render_template(get_template("index"))



@app.route('/audio_player')
def audio_player():
    src = request.args.get('src')
    if not src:
        return "No source specified", 400
    return render_template('audio_player.html', src=src)

@app.route("/delete_chat", methods=["DELETE"])
def delete_chat():
    chat_id = request.args.get("chat_id")
    if not chat_id:
        return jsonify({"error": "chat_id required"}), 400
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_sessions WHERE chat_id = %s", (chat_id,))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status": "success", "message": f"Chat {chat_id} deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list_chats", methods=["GET"])
def list_chats():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT chat_id, title, created_at
            FROM chat_sessions
            WHERE user_id = %s
            ORDER BY created_at DESC NULLS LAST
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        all_chats = []
        for r in rows:
            created_at = r[2].isoformat() if r[2] else "N/A"
            all_chats.append({"chat_id": str(r[0]), "title": r[1] or "(untitled)", "created_at": created_at})
        print("list_chats for user_id", user_id, "returned:", all_chats)
        return jsonify(all_chats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
from flask import send_file
from io import BytesIO

@app.route('/stream_tts', methods=['POST'])
def stream_tts():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        voice = data.get("voice", "fable")
        url = "https://api.openai.com/v1/audio/speech"
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "format": "mp3"
        }
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return jsonify({"error": "TTS API failed"}), 500

        return send_file(BytesIO(response.content), mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mp3')
def mp3_page():
    # 1) Load the main talks metadata
    meta_path    = os.path.join(app.root_path, 'static', 'updated_podcast_json1.json')
    summary_path = os.path.join(app.root_path, 'static', 'summaries.json')

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            talks = json.load(f)
    except FileNotFoundError:
        talks = []

    # 2) Load the summaries file
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
    except FileNotFoundError:
        summaries = {}

    # 3) Build a map from just the “Podcast 001” part → description
    desc_by_number = {}
    for key, desc in summaries.items():
        # extract “Podcast 001” (everything up to first space after the number)
        m = re.match(r'^(Podcast\s*\d+)', key)
        if m:
            number = m.group(1).strip()
            # only keep the first if duplicates
            if number not in desc_by_number:
                desc_by_number[number] = desc

    # 4) Merge descriptions into each talk record by matching the `number` field
    for t in talks:
        num = t.get('number', '').strip()
        t['description'] = desc_by_number.get(num, "")
    # 4.1) Remove any items with no real MP3 link
    talks = [
        t for t in talks
        if t.get("mp3_link") 
           and not t["mp3_link"].rstrip('/').endswith("psychedeli_salon_mp3s")
    ]

    # 5) Apply search filter
    q = request.args.get('q', '').strip().lower()
    if q:
        talks = [t for t in talks if q in t.get('name', '').lower()]

    # 6) Sort by podcast number
    def num_key(t):
        m = re.search(r'(\d+)', t.get('number', ''))
        return int(m.group(1)) if m else float('inf')
    talks.sort(key=num_key)

    # 7) Paginate
    page        = int(request.args.get('page', 1))
    per_page    = 20
    total       = len(talks)
    total_pages = (total + per_page - 1) // per_page
    start       = (page - 1) * per_page
    page_talks  = talks[start:start + per_page]

    # 8) Render
    return render_template(
        get_template("mp3"),
        talks       = page_talks,
        page        = page,
        total_pages = total_pages,
        query       = q
    )







if __name__ == '__main__':
    ensure_collection_exists_with_hnsw()
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)