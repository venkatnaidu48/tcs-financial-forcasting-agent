from ..utils.embeddings import embed_texts
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain import OpenAI, LLMChain, PromptTemplate
import os
OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)

class SimpleRAG:
    def __init__(self):
        self.texts = []
        self.vectors = None
        self.index = None

    def add_documents(self, docs):
        """
        docs: list of {"title":..., "text":...}
        """
        self.texts.extend([d["text"] for d in docs])
       
        vectors = embed_texts(self.texts)
        self.vectors = vectors
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def query(self, q, top_k=3):
        qv = embed_texts([q])[0]
        D, I = self.index.search(np.array([qv]), top_k)
        results = []
        for idx in I[0]:
            results.append(self.texts[idx])
        return results

def extract_themes_and_sentiment(doc_snippets):
    """
    Use LLM to synthesize common themes/sentiment across snippets.
    """
    if not OPENAI_KEY:
        
        vect = TfidfVectorizer(stop_words="english", max_features=20)
        vect.fit(doc_snippets)
        terms = vect.get_feature_names_out().tolist()
        return {"themes": terms[:8], "sentiment": "neutral", "method": "tfidf-fallback"}

    llm = OpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = PromptTemplate(
        input_variables=["snippets"],
        template=(
            "You are an analyst. Given these excerpts from earnings call transcripts, list the 5 recurring themes (short) "
            "and a one-sentence summary of management sentiment (positive / cautious / negative) and any explicit forward-looking statements. "
            "Return JSON: {{'themes': [...], 'sentiment': '...', 'forward_looking': [...]}}.\n\nSnippets:\n{snippets}\n\nJSON:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run({"snippets": "\n\n".join(doc_snippets[:6])})
    import re, json
    m = re.search(r"(\{.*\})", out, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            return {"themes": [], "sentiment": "unknown", "method": "llm-unparseable"}
    return {"themes": [], "sentiment": "unknown", "method": "llm-nojson"}

class QualitativeAnalysisTool:
    def __init__(self):
        self.rag = SimpleRAG()

    def ingest_transcripts(self, transcript_docs):
        """
        transcript_docs: list of {"title":"...", "text":"..."}
        """
        self.rag.add_documents(transcript_docs)

    def analyze(self, query):
        snippets = self.rag.query(query, top_k=4)
        synth = extract_themes_and_sentiment(snippets)
        return {"snippets": snippets, "synthesis": synth}
