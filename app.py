from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import secrets
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

from flask_cors import CORS

from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state, reset_query_state
from tools.pdf_loader import process_pdf
from tools.vector_store import get_or_create_vectorstore

from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------
# Load environment variables
# --------------------------------------
load_dotenv()

# --------------------------------------
# Flask app setup
# --------------------------------------
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load local embedding model
print("Loading local embedding model... (this will take a few seconds the first time)")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully ✅")

app.secret_key = secrets.token_hex(32)

# --------------------------------------
# MongoDB setup
# --------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["caremate"]
sessions_collection = db["sessions"]
messages_collection = db["messages"]

# --------------------------------------
# Global workflow and conversation state
# --------------------------------------
workflow_app = None
conversation_states = {}


# --------------------------------------
# Doctor recommendation functions
# --------------------------------------

def get_embedding(text):
    """Generate embedding vector for given text using local SentenceTransformer"""
    return np.array(embedder.encode(text, normalize_embeddings=True))


def find_related_doctors(user_message, limit=3):
    print("Started")
    """Find doctors semantically related to user symptoms using local embeddings"""
    doctors = list(db["users"].find({"role": "doctor"}))
    print(doctors)
    if not doctors:
        return []

    user_emb = get_embedding(user_message)
    similarities = []

    for doc in doctors:
        profile = doc.get("doctorProfile", {})
        print(profile)
        specialization = profile.get("specialization", "")
        print(specialization)
        qualifications = ", ".join(profile.get("qualifications", []))
        years_exp = profile.get("yearsExperience", "")
        
        doc_text = f"{specialization} {qualifications} {years_exp} years experience"
        doc_emb = doc.get("embedding")

        # If embedding not present, create and save it
        if not doc_emb:
            doc_emb = get_embedding(doc_text)
            db["users"].update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": doc_emb.tolist()}}
            )
        else:
            doc_emb = np.array(doc_emb)

        similarity = np.dot(user_emb, doc_emb)
        similarities.append((doc, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = [d[0] for d in similarities[:limit]]

    return [{
        "name": d.get("name", ""),
        "email": d.get("email", ""),
        "specialization": d.get("doctorProfile", {}).get("specialization", ""),
        "yearsExperience": d.get("doctorProfile", {}).get("yearsExperience", ""),
        "qualifications": d.get("doctorProfile", {}).get("qualifications", []),
    } for d in top_matches]


# --------------------------------------
# MongoDB helper functions
# --------------------------------------

def save_message(session_id, role, content, source=None):
    """Save user or assistant message to MongoDB"""
    messages_collection.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "source": source,
        "timestamp": datetime.now(timezone.utc)
    })
    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            "$set": {"last_active": datetime.now(timezone.utc)}
        },
        upsert=True
    )



def get_chat_history(session_id):
    """Retrieve ordered chat messages for a session"""
    messages = list(messages_collection.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1))
    return messages


def get_all_sessions():
    """Fetch all sessions with preview"""
    sessions = []
    for s in sessions_collection.find().sort("last_active", -1):
        first_user_msg = messages_collection.find_one(
            {"session_id": s["session_id"], "role": "user"},
            sort=[("timestamp", 1)]
        )
        preview = None
        if first_user_msg and first_user_msg.get("content"):
            content = first_user_msg["content"]
            preview = content[:50] + "..." if len(content) > 50 else content

        sessions.append({
            "session_id": s["session_id"],
            "created_at": s.get("created_at"),
            "last_active": s.get("last_active"),
            "preview": preview
        })
    return sessions


def delete_session(session_id):
    """Delete all messages + session document"""
    messages_collection.delete_many({"session_id": session_id})
    sessions_collection.delete_one({"session_id": session_id})


# --------------------------------------
# MediGenius Initialization
# --------------------------------------
def initialize_system():
    global workflow_app

    pdf_path = './data/medical_book.pdf'
    persist_dir = './medical_db/'

    print("Initializing MediGenius System...")

    # Initialize vector DB
    existing_db = get_or_create_vectorstore(persist_dir=persist_dir)

    if not existing_db and os.path.exists(pdf_path):
        print("Creating vector database from PDF...")
        doc_splits = process_pdf(pdf_path)
        get_or_create_vectorstore(documents=doc_splits, persist_dir=persist_dir)
    elif not existing_db:
        print("No vector database and no PDF found — RAG features will be limited")

    workflow_app = create_workflow()
    print("MediGenius Web Interface Ready!")


# --------------------------------------
# Flask Routes
# --------------------------------------

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    global workflow_app, conversation_states

    data = request.json
    message = data.get('message', '')
    session_id = session.get('session_id')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    if not workflow_app:
        return jsonify({'error': 'System not initialized'}), 500

    # Save user message
    save_message(session_id, 'user', message)

    # Fetch last 5 messages (for context)
    previous_messages = list(messages_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(5))
    previous_messages.reverse()  # so oldest comes first

    # Build context string
    context = ""
    for msg in previous_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    # Initialize or get conversation state
    if session_id not in conversation_states:
        conversation_states[session_id] = initialize_conversation_state()

    conversation_state = conversation_states[session_id]
    conversation_state = reset_query_state(conversation_state)

    # Add both context and question
    conversation_state["context"] = context.strip()
    conversation_state["question"] = message

    # Process query through workflow
    result = workflow_app.invoke(conversation_state)
    conversation_states[session_id].update(result)
    related_doctors = find_related_doctors(message)

    # Extract response and source
    response = result.get('generation', 'Unable to generate response.')
    source = result.get('source', 'Unknown')

    # Save assistant response
    save_message(session_id, 'assistant', response, source)

    timestamp = datetime.now().strftime("%I:%M %p")

    return jsonify({
        'response': response,
        'source': source,
        'timestamp': timestamp,
        'related_doctors': related_doctors,
        'context_used': context,  # for debugging (optional)
        'success': bool(result.get('generation'))
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return chat history for the active session"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'messages': []})
    messages = get_chat_history(session_id)
    return jsonify({'messages': messages, 'success': True})


@app.route('/api/sessions', methods=['GET'])
def get_sessions_route():
    """Return all sessions"""
    sessions = get_all_sessions()
    return jsonify({'sessions': sessions, 'success': True})


@app.route('/api/session/<session_id>', methods=['GET'])
def load_session(session_id):
    """Load messages from a specific session"""
    session['session_id'] = session_id
    messages = get_chat_history(session_id)
    return jsonify({
        'messages': messages,
        'session_id': session_id,
        'success': True
    })


@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete a specific session"""
    delete_session(session_id)
    if session.get('session_id') == session_id:
        session['session_id'] = str(uuid.uuid4())
    return jsonify({'message': 'Session deleted', 'success': True})


@app.route('/api/clear', methods=['POST'])
def clear():
    """Reset conversation state (in-memory only)"""
    session_id = session.get('session_id')
    if session_id in conversation_states:
        conversation_states[session_id] = initialize_conversation_state()
    return jsonify({'message': 'Conversation cleared', 'success': True})


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Create a new session"""
    new_session_id = str(uuid.uuid4())
    session['session_id'] = new_session_id
    return jsonify({
        'message': 'New chat created',
        'session_id': new_session_id,
        'success': True
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'MediGenius'})


# --------------------------------------
# Run app
# --------------------------------------
if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=8000)
