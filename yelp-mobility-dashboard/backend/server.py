import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
import json as pyjson

LAST_PICK = {}  # user_id -> business_id

def parse_rag_text(text: str) -> Dict[str, Any]:
    def grab(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    city, state = "", ""
    m = re.search(r"City:\s*([^,]+),\s*([A-Z]{2})\.", text)
    if m:
        city, state = m.group(1).strip(), m.group(2).strip()

    return {
        "name": grab(r"Name:\s*(.*?)\.\s*Categories:"),
        "categories": grab(r"Categories:\s*(.*?)\.\s*City:"),
        "city": city,
        "state": state,
        "rating": grab(r"Rating:\s*([0-9.]+)\s*stars"),
        "review_count": grab(r"Review Count:\s*([0-9]+)"),
    }

def get_topk_with_scores(vectorstore, query: str, k: int = 5):
    # FAISS supports similarity_search_with_score
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in docs_and_scores:
        row = parse_rag_text(doc.page_content)
        row["score"] = float(score)
        # if you store business_id in metadata later, include it here
        out.append(row)
    return out

def safe_json_loads(raw: str):
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    return pyjson.loads(s)

# 1. SETUP & CONFIGURATION
os.environ["GOOGLE_API_KEY"] = "AIzaSyDg1rLmGmu9DERU7cabgg1hm2A1focDCLo" # Uncomment if not set in system env
INDEX_FOLDER = "faiss_index_store_local"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD DATA & BUILD VECTOR STORE
print("Initializing server...")

# A. Load User Profiles (Using your specific path)
# We use r"" (raw string) to handle Windows backslashes correctly
user_profiles_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\user_profiles.json"

try:
    with open(user_profiles_path, "r") as f:
        user_profiles = json.load(f)
    print(f"Loaded {len(user_profiles)} user profiles.")
except FileNotFoundError:
    print(f"WARNING: Could not find user_profiles.json at {user_profiles_path}")
    user_profiles = {}

try:
    # 1. Read the CSV
    csv_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\restaurant_rag_data.csv"
    df = pd.read_csv(csv_path)

    print(f"Processing {len(df)} restaurants...")

    # 3. Use hugging face embeddings
    texts = df["rag_text"].dropna().astype(str).tolist()
    print("Total rows:", len(texts))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [
        Document(
            page_content=str(r["rag_text"]),
            metadata={"business_id": str(r["business_id"]), "name": str(r["name"])}
        )
        for _, r in df.dropna(subset=["rag_text"]).iterrows()
    ]

    if os.path.exists(INDEX_FOLDER):
        vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
    else:
        batch_size = 1000
        vectorstore = FAISS.from_documents(docs[:batch_size], embedding=embeddings)
        for i in range(batch_size, len(docs), batch_size):
            vectorstore.add_documents(docs[i:i+batch_size])
        vectorstore.save_local(INDEX_FOLDER)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("Vector index built successfully!")
    
except Exception as e:
    print(f"CRITICAL ERROR loading restaurant data: {e}")
    retriever = None

# Helper function to format retrieved docs
def format_docs(docs, max_chars=3500):
    lines = []
    for d in docs:
        bid = d.metadata.get("business_id", "UNKNOWN_ID")
        name = d.metadata.get("name", "UNKNOWN_NAME")
        lines.append(f"[{bid}] {name}\n{d.page_content}")
    joined = "\n\n---\n\n".join(lines)
    return joined[:max_chars]

# 3. DEFINE THE RAG CHAIN
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template = """
You are a Restaurant Recommendation Assistant.

RULES:
- You MUST recommend exactly ONE restaurant from the CANDIDATES list.
- Output MUST be valid JSON with keys: business_id, name, reason.
- If you cannot decide, pick the best match from CANDIDATES (do NOT invent new places).

USER HISTORY:
{user_history}

CANDIDATES (choose from these only):
{context}

USER QUESTION:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

class ChatRequest(BaseModel):
    message: str
    user_id: str

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     if not retriever:
#         raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

#     # 1. Look up user history
#     user_history = user_profiles.get(request.user_id, "No past history available (New User).")
    
#     # 2. Define the chain
#     chain = (
#         {
#             "context": retriever | format_docs, 
#             "question": RunnablePassthrough(),
#             "user_history": lambda x: user_history 
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     # 3. Invoke
#     try:
#         response = chain.invoke(request.message)
#         return {"reply": response}
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

    user_history = user_profiles.get(request.user_id, "No past history available (New User).")

    # 1) Retrieve candidates with scores (verifiable)
    search_query = f"{request.message}\n\nUser history:\n{user_history}"
    docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=8)
    candidate_docs = [d for d, s in docs_and_scores]

    last = LAST_PICK.get(request.user_id)
    if last:
        candidate_docs = [d for d in candidate_docs if d.metadata.get("business_id") != last]

    candidate_ids = {d.metadata.get("business_id") for d in candidate_docs}

    context = format_docs(candidate_docs)

    # 2) Call LLM
    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "user_history": lambda _: user_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke(request.message)

    # 3) Verify JSON + groundedness
    parsed = None
    grounded = False
    try:
        parsed = safe_json_loads(raw)
        grounded = parsed.get("business_id") in candidate_ids
    except Exception:
        grounded = False

    # 4) Fallback if not grounded
    if not grounded:
        top = candidate_docs[0]
        parsed = {
            "business_id": top.metadata.get("business_id"),
            "name": top.metadata.get("name"),
            "reason": "Fallback: model output was not grounded; returning top retrieved candidate."
        }
        LAST_PICK[request.user_id] = parsed["business_id"]

    # 5) Return with evidence (so you can verify in UI)
    evidence = []
    for d, score in docs_and_scores:
        row = parse_rag_text(d.page_content)
        row.update({
            "business_id": d.metadata.get("business_id"),
            "name_meta": d.metadata.get("name"),
            "score": float(score),
        })
        evidence.append(row)

    return {
        "reply": parsed,
        "is_grounded": grounded,
        "evidence": evidence
    }

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 instead of 0.0.0.0 for better Windows compatibility
    uvicorn.run(app, host="127.0.0.1", port=8000)