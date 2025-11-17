import os
import re
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

import openai
from ..utils.embeddings import embed_texts

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

import chromadb
client = chromadb.Client()


class QualitativeAnalysisTool:
    """RAG tool using ChromaDB + SentenceTransformers embeddings."""

    def __init__(self, collection_name="tcs_transcripts"):
        self.collection_name = collection_name

        try:
            self.collection = client.get_collection(collection_name)
        except Exception:
            self.collection = client.create_collection(name=collection_name)

    def ingest_transcripts(self, transcript_docs: List[Dict[str, str]]):
        """Add transcript texts into Chroma."""
        if not transcript_docs:
            return

        texts = [doc["text"] for doc in transcript_docs]
        ids = [doc.get("title", f"doc_{i}") for i, doc in enumerate(transcript_docs)]
        embeddings = embed_texts(texts).tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings
        )

    def query(self, q: str, top_k: int = 4) -> List[str]:
        """Retrieve top transcript chunks."""
        if not q:
            return []

        q_emb = embed_texts([q])[0].tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )

        if "documents" in results and results["documents"]:
            return results["documents"][0]

        return []


def extract_themes_and_sentiment(snippets: List[str]) -> Dict:
    """Use OpenAI to extract themes + sentiment as JSON."""

    if not snippets:
        return {"themes": [], "sentiment": "neutral", "forward_looking": []}

    if not OPENAI_KEY:
        return {
            "themes": ["no_openai_key"],
            "sentiment": "neutral",
            "forward_looking": []
        }

    prompt = (
        "You are an expert analyst.\n"
        "Based on the transcript excerpts below, identify:\n"
        "- 3 to 5 major recurring themes\n"
        "- sentiment (positive, cautious, or negative)\n"
        "- forward-looking statements\n"
        "Return ONLY valid JSON.\n\n"
        f"EXCERPTS:\n{snippets}\n\nJSON:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400
        )
        out = response["choices"][0]["message"]["content"]
    except Exception as e:
        print("OpenAI error:", e)
        return {"themes": [], "sentiment": "unknown", "forward_looking": []}

    # Attempt to parse JSON
    match = re.search(r"(\{.*\})", out, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return {"themes": [], "sentiment": "unknown", "forward_looking": []}
