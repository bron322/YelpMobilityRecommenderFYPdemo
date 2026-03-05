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
from pydantic import BaseModel
import difflib

LAST_PICK = {}  # user_id -> business_id

class ValidateUserRequest(BaseModel):
    user_id: str

def normalize_uid(uid: str) -> str:
    uid = (uid or "").strip()
    uid = uid.strip('"').strip("'")
    # handle “smart dash” (—, –) copied from some UIs
    uid = uid.replace("\u2014", "-").replace("\u2013", "-")
    return uid

def resolve_user_id(uid: str) -> str | None:
    # 1) exact
    if uid in user_profiles:
        return uid

    # 2) match ignoring leading - and _ (your IDs often start with --- or -_)
    needle = uid.lstrip("-_")
    for k in user_profiles.keys():
        if k.lstrip("-_") == needle:
            return k

    # 3) last resort: endswith (if user pasted a suffix)
    for k in user_profiles.keys():
        if k.endswith(needle):
            return k

    return None
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
user_profiles_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\user_profiles_enriched.json"

def load_json_with_fallback(path: str):
    # Try common encodings on Windows
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_err = e
    raise last_err

try:
    user_profiles = load_json_with_fallback(user_profiles_path)
    print(f"Loaded {len(user_profiles)} user profiles.")
except FileNotFoundError:
    print(f"WARNING: Could not find user_profiles.json at {user_profiles_path}")
    user_profiles = {}

try:
    # 1. Read the CSV
    csv_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\restaurant_rag_data.csv"
    df = pd.read_csv(csv_path)

    def extract_city_state(rag_text: str):
        m = re.search(r"City:\s*([^,]+),\s*([A-Z]{2})\.", rag_text)
        if not m:
            return None, None
        return m.group(1).strip(), m.group(2).strip()
    
    df["city"], df["state"] = zip(*df["rag_text"].fillna("").map(extract_city_state))
    print(f"Processing {len(df)} restaurants...")

    def infer_user_city_state_from_history(user_history: str):
        pairs = re.findall(r"([A-Za-z .'-]+),\s*([A-Z]{2})", user_history)
        if not pairs:
            return None
        # count frequency
        from collections import Counter
        c = Counter([(a.strip(), b.strip()) for a,b in pairs])
        return c.most_common(1)[0][0]  # (city, state)
    
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

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     if not retriever:
#         raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

#     user_history = user_profiles.get(request.user_id, "No past history available (New User).")

#     # 1) Retrieve candidates with scores (verifiable)
#     search_query = f"{request.message}\n\nUser history:\n{user_history}"
#     docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=50)

#     user_loc = infer_user_city_state_from_history(user_history)  # (city,state) or None

#     filtered = []
#     for d, s in docs_and_scores:
#         row = parse_rag_text(d.page_content)
#         if user_loc:
#             if (row["city"], row["state"]) != user_loc:
#                 continue
#         filtered.append((d, s))

#     # fallback if too few local
#     if len(filtered) < 8 and user_loc:
#         # relax to same state
#         filtered = [(d,s) for d,s in docs_and_scores
#                     if parse_rag_text(d.page_content)["state"] == user_loc[1]]

#     if len(filtered) < 8:
#         filtered = docs_and_scores

#     candidate_docs = [d for d, s in filtered[:8]]

#     last = LAST_PICK.get(request.user_id)
#     if last:
#         candidate_docs = [d for d in candidate_docs if d.metadata.get("business_id") != last]

#     candidate_ids = {d.metadata.get("business_id") for d in candidate_docs}

#     context = format_docs(candidate_docs)

#     # 2) Call LLM
#     chain = (
#         {
#             "context": lambda _: context,
#             "question": RunnablePassthrough(),
#             "user_history": lambda _: user_history,
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     raw = chain.invoke(request.message)

#     # 3) Verify JSON + groundedness
#     parsed = None
#     grounded = False
#     try:
#         parsed = safe_json_loads(raw)
#         grounded = parsed.get("business_id") in candidate_ids
#     except Exception:
#         grounded = False

#     # 4) Fallback if not grounded
#     if not grounded:
#         top = candidate_docs[0]
#         parsed = {
#             "business_id": top.metadata.get("business_id"),
#             "name": top.metadata.get("name"),
#             "reason": "Fallback: model output was not grounded; returning top retrieved candidate."
#         }
#         LAST_PICK[request.user_id] = parsed["business_id"]

#     # 5) Return with evidence (so you can verify in UI)
#     evidence = []
#     for d, score in docs_and_scores:
#         row = parse_rag_text(d.page_content)
#         row.update({
#             "business_id": d.metadata.get("business_id"),
#             "name_meta": d.metadata.get("name"),
#             "score": float(score),
#         })
#         evidence.append(row)

#     return {
#         "reply": parsed,
#         "is_grounded": grounded,
#         "evidence": evidence
#     }

from typing import Optional, Tuple

def get_user_profile(user_id: str) -> dict:
    p = user_profiles.get(user_id)
    return p if isinstance(p, dict) else {}

def get_doc_city_state(doc) -> Tuple[Optional[str], Optional[str]]:
    # If you later add city/state into doc.metadata, this will use it automatically
    city = doc.metadata.get("city")
    state = doc.metadata.get("state")
    if city and state:
        return city, state

    row = parse_rag_text(doc.page_content)
    return (row.get("city") or None, row.get("state") or None)

def filter_by_location(docs_and_scores, user_loc: Optional[Tuple[str, str]], min_k: int = 8):
    """
    Location-first relaxation:
      1) same city+state
      2) same state
      3) global
    Returns: (filtered_docs_and_scores, location_mode)
    """
    if not user_loc:
        return docs_and_scores, "global"

    u_city, u_state = user_loc

    # 1) same city+state
    city_filtered = []
    for d, s in docs_and_scores:
        city, state = get_doc_city_state(d)
        if city == u_city and state == u_state:
            city_filtered.append((d, s))
    if len(city_filtered) >= min_k:
        return city_filtered, "city"

    # 2) same state
    state_filtered = []
    for d, s in docs_and_scores:
        _, state = get_doc_city_state(d)
        if state == u_state:
            state_filtered.append((d, s))
    if len(state_filtered) >= min_k:
        return state_filtered, "state"

    # 3) global
    return docs_and_scores, "global"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

    # --- 0) Read enriched profile correctly ---
    profile = get_user_profile(request.user_id)

    user_history_text = profile.get("history_text", "No past history available (New User).")
    top_city = profile.get("top_city")
    top_state = profile.get("top_state")
    user_loc = (top_city, top_state) if top_city and top_state else None

    # --- 1) Retrieve globally, then gate by location ---
    search_query = f"{request.message}\n\nUser history:\n{user_history_text}"
    docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=50)

    filtered, location_mode = filter_by_location(docs_and_scores, user_loc, min_k=8)

    # --- 2) Build candidate list (exclude last pick) ---
    candidate_docs = [d for d, _ in filtered]
    last = LAST_PICK.get(request.user_id)
    if last:
        candidate_docs = [d for d in candidate_docs if d.metadata.get("business_id") != last]

    candidate_docs = candidate_docs[:8]
    candidate_ids = {d.metadata.get("business_id") for d in candidate_docs}

    context = format_docs(candidate_docs)

    # --- 3) LLM call ---
    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "user_history": lambda _: user_history_text,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    raw = chain.invoke(request.message)

    # --- 4) Verify JSON + groundedness ---
    grounded = False
    try:
        parsed = safe_json_loads(raw)
        grounded = parsed.get("business_id") in candidate_ids
    except Exception:
        parsed = None
        grounded = False

    # --- 5) Fallback if not grounded ---
    if not grounded and candidate_docs:
        top = candidate_docs[0]
        parsed = {
            "business_id": top.metadata.get("business_id"),
            "name": top.metadata.get("name"),
            "reason": "Fallback: model output was not grounded; returning top retrieved candidate."
        }
        LAST_PICK[request.user_id] = parsed["business_id"]

    # --- 6) Evidence for UI/debug (still global list) ---
    # evidence should match the candidate list shown to the LLM
    evidence = []
    for d in candidate_docs:
        row = parse_rag_text(d.page_content)
        row.update({
            "business_id": d.metadata.get("business_id"),
            "name_meta": d.metadata.get("name"),
            "city": row.get("city"),
            "state": row.get("state"),
        })
        evidence.append(row)

    return {
        "reply": parsed,
        "is_grounded": grounded,
        "location_mode": location_mode,  # "city" | "state" | "global"
        "inferred_location": {"city": top_city, "state": top_state} if user_loc else None,
        "evidence": evidence,
    }

@app.post("/validate_user")
async def validate_user(req: ValidateUserRequest):
    raw = req.user_id
    uid = normalize_uid(raw)

    found = resolve_user_id(uid)
    if not found:
        # helpful suggestions
        keys = list(user_profiles.keys())
        suggestions = difflib.get_close_matches(uid, keys, n=5, cutoff=0.3)
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown user_id",
                "received": uid,
                "tip": "Paste the FULL id including leading dashes/underscores (e.g. ---zema...)",
                "suggestions": suggestions,
            },
        )

    profile = user_profiles[found]
    preview = profile.get("history_text", "")[:300] if isinstance(profile, dict) else str(profile)[:300]
    return {"ok": True, "user_id": found, "preview": preview}

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 instead of 0.0.0.0 for better Windows compatibility
    uvicorn.run(app, host="127.0.0.1", port=8000)