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

# public GCS bucket (same default you used elsewhere)
GCS_BUCKET = os.getenv("BUCKET_NAME", "psychedeli_salon_mp3s")


# Load PostgreSQL connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize Flask
app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set your OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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

def call_gpt4o_mini_model(prompt, user_id, chat_id=None, relevant_context=None, custom_instructions=""):
    """Calls GPT, returns the model's text response along with chat_id."""
    try:
        if relevant_context is None:
            relevant_context = search_similar_with_hnsw(prompt)
        if chat_id is None:
            chat_id = str(uuid.uuid4())

        global conversation_contexts
        if user_id not in conversation_contexts:
            conversation_contexts[user_id] = []

        conversation_history = conversation_contexts[user_id]
        # Keep only the last 10 exchanges
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
            conversation_contexts[user_id] = conversation_history

        combined_instructions = ('''You are the AI embodiment of the psychedelic salons  ideas and works. You are being fed 
                                 transcribed snippets from over 700 of ther psychedelis salon's talks and I want you to use the context in the snippets, along with your 
                                 own knowledge, to answer user's queries. Do not mention the snippets directly. Instead, occasionally quote the speakers in long quotes,
                                 and speak in a way that appeals to fans of their work. Do not begin your response with repetitive, overly dramatic phrases like "AHH". Please don't fluff up the repsonses.
                                 Do not use lists unless the user asks for them. Your goal is to deliver mind-blowing responses that unpack complex ideas, challenging cultural and social norms.
                                 Please speak very articulately, as if you are the speaker themselves—using their style of combining concepts into new, meaningful insights.
                                 If it helps the user, recommend a talk with a link at the end of your response (using only metadata links from the snippets).
                                 Make responses long enough to cover the query very well, but not so long as to be burdonsome to read.
                                 ''')
        if custom_instructions.strip():
            combined_instructions += f"\n\nCustom Instructions:\n{custom_instructions}"

        # Always prepend the system message in every call.
        messages = [{"role": "system", "content": combined_instructions}]
        # Append any conversation history if exists.
        for exchange in conversation_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        # Append current query with context.
        messages.append({
            "role": "user",
            "content": f"Context:\n{relevant_context}\n\nPrompt:\n{prompt}"
        })

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        model_response = response['choices'][0]['message']['content'].strip()
        conversation_history.append({"user": prompt, "assistant": model_response})
        conversation_contexts[user_id] = conversation_history
        return model_response, chat_id

    except openai.error.OpenAIError as api_err:
        return f"❌ OpenAI API error: {str(api_err)}", chat_id
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", chat_id


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

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        data = request.json
        user_input = data.get('user_input', '')
        custom_instructions = data.get('custom_instructions', '')
        chat_id = data.get('chat_id')
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'user_id is required'}), 400
        if not user_input:
            return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

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

        ai_response, updated_chat_id = call_gpt4o_mini_model(
            prompt=user_input,
            user_id=user_id,
            chat_id=chat_id,
            custom_instructions=custom_instructions
        )
        chat_id = updated_chat_id or chat_id

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

@app.route('/mp3')
def mp3_page():
    q       = request.args.get('q', '').strip().lower()
    page    = int(request.args.get('page', 1))
    per_page= 20

    # 1) Load summaries.json
    summaries_path = os.path.join(app.root_path, 'static', 'summaries.json')
    try:
        with open(summaries_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)  # { title: description, ... }
    except Exception as e:
        print(f"❌ Error loading summaries.json: {e}")
        summaries = {}

    # 2) Load updated JSON (with working mp3_link fields)
    updated_path = os.path.join(app.root_path, 'static', 'updated_podcast_json1.json')
    try:
        with open(updated_path, 'r', encoding='utf-8') as f:
            updated = json.load(f)    # [ { title, mp3_link, ... }, ... ]
    except Exception as e:
        print(f"❌ Error loading updated_podcast_json1.json: {e}")
        updated = []

    # 3) Build combined list
    talks = []
    for e in updated:
        title    = e.get('title','').strip()
        mp3_link = e.get('mp3_link','').strip()
        desc     = summaries.get(title, '')

        if q and q not in title.lower():
            continue

        talks.append({
            'title':       title,
            'description': desc,
            'mp3_link':    mp3_link
        })

    # 4) Sort by numeric podcast ID in title
    def extract_num(t):
        m = re.search(r'(\d+)', t['title'])
        return int(m.group(1)) if m else float('inf')
    talks.sort(key=extract_num)

    # 5) Paginate
    total       = len(talks)
    total_pages = (total + per_page - 1) // per_page
    start, end  = (page-1)*per_page, page*per_page
    page_talks  = talks[start:end]

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