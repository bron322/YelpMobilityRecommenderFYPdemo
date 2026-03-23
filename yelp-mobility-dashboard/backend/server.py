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
from typing import Literal
from typing import Optional, Tuple
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

STEP2_MOBILITY_TABLE_PATH = r"C:\Users\lebro\OneDrive - Nanyang Technological University\Github\YelpFYP\step2_outputs\user_mobility_table.csv"
STEP2_USER_HUBS_PATH = r"C:\Users\lebro\OneDrive - Nanyang Technological University\Github\YelpFYP\step2_outputs\user_hubs.csv"
STEP3_CANDIDATES_PATH = r"C:\Users\lebro\OneDrive - Nanyang Technological University\Github\YelpFYP\step3_outputs\step3_candidates.parquet"
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from backend/.env")
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

def load_csv_with_fallback(path: str, **kwargs):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
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
            id=str(r["business_id"]),
            page_content=str(r["rag_text"]),
            metadata={
                "business_id": str(r["business_id"]),
                "name": str(r["name"]),
                "city": None if pd.isna(r["city"]) else str(r["city"]),
                "state": None if pd.isna(r["state"]) else str(r["state"]),
            }
        )
        for _, r in df.dropna(subset=["rag_text"]).iterrows()
    ]

    doc_by_business_id = {
        d.metadata["business_id"]: d
        for d in docs
    }

    REBUILD_FAISS = False

    if os.path.exists(INDEX_FOLDER) and not REBUILD_FAISS:
        vectorstore = FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        batch_size = 1000
        vectorstore = FAISS.from_documents(docs[:batch_size], embedding=embeddings)
        for i in range(batch_size, len(docs), batch_size):
            vectorstore.add_documents(docs[i:i+batch_size])
        vectorstore.save_local(INDEX_FOLDER)

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("Vector index built successfully!")
    try:
        print("Loading mobility table...")
        mobility_df = load_csv_with_fallback(STEP2_MOBILITY_TABLE_PATH, dtype={"user_id": str})

        print("Loading user hubs...")
        user_hubs_df = load_csv_with_fallback(STEP2_USER_HUBS_PATH, dtype={"user_id": str})

        print("Loading step3 candidates parquet...")
        step3_candidates_df = pd.read_parquet(STEP3_CANDIDATES_PATH)

        # Normalise IDs
        mobility_df["user_id"] = mobility_df["user_id"].astype(str)
        user_hubs_df["user_id"] = user_hubs_df["user_id"].astype(str)
        step3_candidates_df["user_id"] = step3_candidates_df["user_id"].astype(str)

        # Adjust this if your parquet uses a different business-id column name
        if "business_id" not in step3_candidates_df.columns:
            if "candidate_business_id" in step3_candidates_df.columns:
                step3_candidates_df = step3_candidates_df.rename(
                    columns={"candidate_business_id": "business_id"}
                )
            else:
                raise ValueError(
                    f"step3_candidates.parquet has no business_id column. Columns: {step3_candidates_df.columns.tolist()}"
                )

        step3_candidates_df["business_id"] = step3_candidates_df["business_id"].astype(str)

        # Optional: preserve ranking if you have a rank column
        if "rank" in step3_candidates_df.columns:
            step3_candidates_df = step3_candidates_df.sort_values(["user_id", "rank"])

        mobility_lookup = mobility_df.set_index("user_id").to_dict("index")

        user_hubs_lookup = (
            user_hubs_df.groupby("user_id")
            .apply(lambda g: g.to_dict("records"))
            .to_dict()
        )

        step3_candidate_lookup = (
            step3_candidates_df.groupby("user_id")["business_id"]
            .apply(lambda s: list(dict.fromkeys(s.tolist())))
            .to_dict()
        )

        print(f"Loaded mobility profiles for {len(mobility_lookup)} users.")
        print(f"Loaded user hubs for {len(user_hubs_lookup)} users.")
        print(f"Loaded Step 3 candidate pools for {len(step3_candidate_lookup)} users.")

    except Exception as e:
        print(f"WARNING: Could not load Step 2/3 artifacts: {e}")
        mobility_lookup = {}
        user_hubs_lookup = {}
        step3_candidate_lookup = {}
    
except Exception as e:
    print(f"CRITICAL ERROR loading restaurant data: {e}")
    retriever = None
    vectorstore = None
    doc_by_business_id = {}
    mobility_lookup = {}
    user_hubs_lookup = {}
    step3_candidate_lookup = {}

# Helper function to format retrieved docs
def format_docs(docs, max_chars=3500):
    lines = []
    for d in docs:
        bid = d.metadata.get("business_id", "UNKNOWN_ID")
        name = d.metadata.get("name", "UNKNOWN_NAME")
        lines.append(f"[{bid}] {name}\n{d.page_content}")
    joined = "\n\n---\n\n".join(lines)
    return joined[:max_chars]

POOL_INDEX_CACHE = {}

def get_user_mobility_info(user_id: str) -> dict:
    return mobility_lookup.get(user_id, {})

def get_step3_candidate_docs(
    user_id: str,
    mode: str,
    top_city: Optional[str],
    top_state: Optional[str],
):
    business_ids = step3_candidate_lookup.get(user_id, [])
    pool_docs = [
        doc_by_business_id[bid]
        for bid in business_ids
        if bid in doc_by_business_id
    ]

    if not pool_docs:
        return [], "no_step3_pool"

    # Keep your current UX:
    # local = stricter city-only slice of the mobility-aware pool
    # explorer = full mobility-aware pool
    if mode == "local":
        if top_city and top_state:
            city_docs = [
                d for d in pool_docs
                if (d.metadata.get("city"), d.metadata.get("state")) == (top_city, top_state)
            ]
            if city_docs:
                return city_docs, "step3_city_pool"
            return [], "step3_city_none"
        return [], "no_user_location"

    return pool_docs, "step3_mobility_pool"

def get_cached_pool_index(
    user_id: str,
    mode: str,
    top_city: Optional[str],
    top_state: Optional[str],
    pool_docs: List[Document],
):
    cache_key = (user_id, mode, top_city or "", top_state or "")
    if cache_key not in POOL_INDEX_CACHE:
        unique_pool_docs = []
        seen_ids = set()

        for d in pool_docs:
            bid = d.metadata.get("business_id")
            if not bid or bid in seen_ids:
                continue
            seen_ids.add(bid)

            unique_pool_docs.append(
                Document(
                    id=str(bid),
                    page_content=d.page_content,
                    metadata=d.metadata.copy(),
                )
            )

        POOL_INDEX_CACHE[cache_key] = FAISS.from_documents(
            unique_pool_docs,
            embedding=embeddings
        )

    return POOL_INDEX_CACHE[cache_key]

def get_explorer_candidate_docs(
    user_id: str,
    search_query: str,
    top_city: Optional[str],
    top_state: Optional[str],
    k_global: int = 200,
    step3_keep: int = 60,
    global_keep: int = 60,
):
    # taste prior from Step 3
    step3_ids = step3_candidate_lookup.get(user_id, [])
    step3_docs = [
        doc_by_business_id[bid]
        for bid in step3_ids
        if bid in doc_by_business_id
    ]

    # broader semantic retrieval from full corpus
    global_hits = vectorstore.similarity_search_with_score(search_query, k=k_global)
    global_docs = [d for d, _ in global_hits]

    # prefer places outside the user's usual top city when exploring
    if top_city and top_state:
        non_home_docs = [
            d for d in global_docs
            if (d.metadata.get("city"), d.metadata.get("state")) != (top_city, top_state)
        ]
        if len(non_home_docs) >= global_keep:
            global_docs = non_home_docs

    blended = unique_docs_by_business_id(step3_docs[:step3_keep] + global_docs[:global_keep])

    if not blended:
        return [], "explorer_none"

    return blended, "explorer_blended_pool"

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

def filter_by_location(
    docs_and_scores,
    user_loc: Optional[Tuple[str, str]],
    mode: str,
    min_k: int = 8,
):
    # No inferred location → only Explorer can go global
    if not user_loc:
        if mode == "local":
            return [], "no_location"   # strict
        return docs_and_scores, "global"

    u_city, u_state = user_loc

    def doc_city_state(d):
        city = d.metadata.get("city")
        state = d.metadata.get("state")
        if city and state:
            return city, state
        row = parse_rag_text(d.page_content)
        return row.get("city") or None, row.get("state") or None

    city_filtered = [(d, s) for d, s in docs_and_scores if doc_city_state(d) == (u_city, u_state)]
    state_filtered = [(d, s) for d, s in docs_and_scores if doc_city_state(d)[1] == u_state]

    if mode == "local":
        # ✅ STRICT: only city results; never expand to global
        if len(city_filtered) >= 1:
            return city_filtered, "city"
        return [], "city_none"

    # mode == "explorer"
    # ✅ Wider: state if possible, else global
    if len(state_filtered) >= 1:
        return state_filtered, "state"
    return docs_and_scores, "global"

def infer_query_mode(message: str, mobility_label: Optional[str]) -> Tuple[str, str]:
    text = (message or "").lower()

    local_hints = [
        "near me", "nearby", "close by", "close to me",
        "around me", "around here", "in my area",
        "same area", "same city", "my city", "local"
    ]

    explorer_hints = [
        "explore", "something new", "new place", "different place",
        "branch out", "outside my area", "worth travelling",
        "worth traveling", "not my usual", "somewhere else",
        "hidden gem", "adventurous", "surprise me"
    ]

    if any(h in text for h in explorer_hints):
        return "explorer", "prompt_explorer"

    if any(h in text for h in local_hints):
        return "local", "prompt_local"

    # fallback prior from mobility profile
    if mobility_label in {"explorer", "sparse"}:
        return "explorer", "mobility_default"

    return "local", "mobility_default"

def unique_docs_by_business_id(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        bid = d.metadata.get("business_id")
        if bid and bid not in seen:
            seen.add(bid)
            out.append(d)
    return out

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

    resolved_user_id = resolve_user_id(normalize_uid(request.user_id)) or request.user_id
    profile = get_user_profile(resolved_user_id)

    user_history_text = profile.get("history_text", "No past history available (New User).")
    top_city = profile.get("top_city")
    top_state = profile.get("top_state")
    user_loc = (top_city, top_state) if top_city and top_state else None

    mobility_info = get_user_mobility_info(resolved_user_id)
    mobility_label = mobility_info.get("mobility_label")

    effective_mode, intent_source = infer_query_mode(request.message, mobility_label)

    search_query = f"{request.message}\n\nUser history:\n{user_history_text}"

    filtered = []
    pool_docs = []

    if effective_mode == "local":
        pool_docs, retrieval_source = get_step3_candidate_docs(
            resolved_user_id,
            "local",
            top_city,
            top_state,
        )

        if pool_docs:
            allowed_ids = {d.metadata["business_id"] for d in pool_docs}
            global_hits = vectorstore.similarity_search_with_score(search_query, k=200)
            filtered = [(d, s) for d, s in global_hits if d.metadata.get("business_id") in allowed_ids]

            if len(filtered) < 8:
                pool_index = get_cached_pool_index(
                    resolved_user_id,
                    "local",
                    top_city,
                    top_state,
                    pool_docs
                )
                filtered = pool_index.similarity_search_with_score(
                    search_query,
                    k=min(50, len(pool_docs))
                )
        else:
            docs_and_scores = vectorstore.similarity_search_with_score(search_query, k=50)
            filtered, retrieval_source = filter_by_location(
                docs_and_scores,
                user_loc=user_loc,
                mode="local",
                min_k=8,
            )

    else:
        pool_docs, retrieval_source = get_explorer_candidate_docs(
            resolved_user_id,
            search_query,
            top_city,
            top_state,
        )

        if pool_docs:
            pool_index = get_cached_pool_index(
                resolved_user_id,
                "explorer",
                top_city,
                top_state,
                pool_docs
            )
            filtered = pool_index.similarity_search_with_score(
                search_query,
                k=min(50, len(pool_docs))
            )
        else:
            filtered = vectorstore.similarity_search_with_score(search_query, k=50)

    if not filtered:
        return {
            "reply": "I couldn't find enough strong matches for this query. Try asking for something broader, more exploratory, or a different cuisine.",
            "is_grounded": True,
            "location_mode": retrieval_source,
            "inferred_location": {"city": top_city, "state": top_state} if user_loc else None,
            "evidence": [],
            "travel_mode": effective_mode,
            "retriever_version": "mobility_aware_step3_rerank",
            "mobility_label": mobility_label,
            "candidate_pool_size": len(pool_docs),
            "effective_mode": effective_mode,
            "intent_source": intent_source,
        }
    
    # --- 2) Build candidate list (exclude last pick) ---
    scored_candidates = filtered
    last = LAST_PICK.get(resolved_user_id)
    if last:
        scored_candidates = [
            (d, s) for d, s in scored_candidates
            if d.metadata.get("business_id") != last
        ]

    if not scored_candidates:
        scored_candidates = filtered[:8]

    scored_candidates = scored_candidates[:8]
    candidate_docs = [d for d, _ in scored_candidates]
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
    if grounded and parsed:
        LAST_PICK[resolved_user_id] = parsed["business_id"]

    if not grounded and candidate_docs:
        top = candidate_docs[0]
        parsed = {
            "business_id": top.metadata.get("business_id"),
            "name": top.metadata.get("name"),
            "reason": "Fallback: model output was not grounded; returning top retrieved candidate."
        }
        LAST_PICK[resolved_user_id] = parsed["business_id"]

    # --- 6) Evidence for UI/debug (still global list) ---
    # evidence should match the candidate list shown to the LLM
    evidence = []
    for d, score in scored_candidates:
        row = parse_rag_text(d.page_content)
        row.update({
            "business_id": d.metadata.get("business_id"),
            "name_meta": d.metadata.get("name"),
            "city": d.metadata.get("city"),
            "state": d.metadata.get("state"),
            "score": float(score),
        })
        evidence.append(row)

    print("EFFECTIVE MODE:", effective_mode)

    return {
        "reply": parsed,
        "is_grounded": grounded,
        "location_mode": retrieval_source,
        "inferred_location": {"city": top_city, "state": top_state} if user_loc else None,
        "evidence": evidence,
        "travel_mode": effective_mode,
        "retriever_version": "mobility_aware_step3_rerank",
        "mobility_label": mobility_label,
        "candidate_pool_size": len(pool_docs),
        "num_reranked_results": len(filtered),
        "user_hubs": user_hubs_lookup.get(resolved_user_id, [])[:3],  # optional preview
        "effective_mode": effective_mode,
        "intent_source": intent_source,
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