# vector_append.py
import os
import gemini
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup OpenAI API for vector embedding
gemini.api_key = os.getenv("GEMINI_API_KEY")

# Appwrite config
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_DB_ID = os.getenv("APPWRITE_DB_ID")
APPWRITE_COLLECTION_ID = os.getenv("APPWRITE_COLLECTION_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")

headers = {
    "Content-Type": "application/json",
    "X-Appwrite-Project": APPWRITE_PROJECT_ID,
    "X-Appwrite-Key": APPWRITE_API_KEY,
}


def generate_embedding(text):
    try:
        print("[INFO] Generating embedding...")
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"[ERROR] Failed to generate embedding: {e}")
        return None


def append_to_appwrite(role, text, embedding):
    try:
        print("[INFO] Appending data to Appwrite...")
        payload = {
            "data": {
                "role": role,
                "text": text,
                "embedding": embedding
            }
        }
        res = requests.post(
            f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DB_ID}/collections/{APPWRITE_COLLECTION_ID}/documents",
            headers=headers,
            json=payload
        )
        res.raise_for_status()
        print(f"[INFO] Successfully added to Appwrite: {res.json().get('$id')}")
    except Exception as e:
        print(f"[ERROR] Failed to add to Appwrite: {e}")


def process_transcript(role, text):
    embedding = generate_embedding(text)
    if embedding:
        append_to_appwrite(role, text, embedding)
