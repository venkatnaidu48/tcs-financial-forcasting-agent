import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from ..utils.embeddings import embed_texts

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

chroma = chromadb.Client()


class QualitativeAnalysisTool:
    def __init__(self, name="tcs_transcripts"):
        try:
            self.collection = chroma.get_collection(name)
        except:
            self.collection = chroma.create_collection(name)

    def ingest_transcripts(self, docs: List[Dict]):
        texts = [d["text"] for d in docs]
        ids = [d["title"] for d in docs]
        embeddings = embed_texts(texts).tolist()
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def query(self, q: str, k=4):
        q_emb = embed_texts([q])[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        return res["documents"][0] if res["documents"] else []


def extract_themes_and_sentiment(snippets: List[str]) -> Dict:
    if not OPENAI_KEY or not snippets:
        return {"themes": [], "sentiment": "neutral", "forward_looking": []}

    prompt = (
        "Analyze these transcript excerpts.\n"
        "Return JSON with keys: themes, sentiment, forward_looking.\n\n"
        f"{snippets}\n\nJSON:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content
        match = re.search(r"(\{.*\})", text, flags=re.S)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        print("Qualitative LLM error:", e)

    return {"themes": [], "sentiment": "unknown", "forward_looking": []}
