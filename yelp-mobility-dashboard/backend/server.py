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

# Inherit from 'Embeddings' so FAISS recognizes it correctly
class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Return a list of 768 dummy numbers for each text
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 768
    
# 1. SETUP & CONFIGURATION
os.environ["GOOGLE_API_KEY"] = "AIzaSyDg1rLmGmu9DERU7cabgg1hm2A1focDCLo" # Uncomment if not set in system env

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

# B. Load Restaurant Data & Build Vector Index
# INDEX_FOLDER = "faiss_index_store"

# # Initialize Embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # 1. Try to load existing index (FREE & FAST)
# if os.path.exists(INDEX_FOLDER):
#     print("Loading vector index from local disk...")
#     # allow_dangerous_deserialization is needed for local files
#     vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     print("Vector index loaded successfully!")

# # 2. If no index exists, build it (COSTS QUOTA)
# else:
#     print("Index not found. Building new index from CSV...")
#     try:
#         csv_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\restaurant_rag_data.csv"
#         df = pd.read_csv(csv_path)
        
#         # === CRITICAL STEP: LIMIT DATA ===
#         # Use only 50 rows for now so you don't get blocked again.
#         # Once this works, you can increase it later.
#         df = df.head(50) 
#         print(f"Processing {len(df)} restaurants...")

#         vectorstore = FAISS.from_texts(
#             texts=df['rag_text'].tolist(),
#             embedding=embeddings
#         )
        
#         # Save to disk so we don't have to do this again
#         vectorstore.save_local(INDEX_FOLDER)
        
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#         print("Vector index built and SAVED to disk successfully!")
        
#     except Exception as e:
#         print(f"CRITICAL ERROR loading restaurant data: {e}")
#         retriever = None

# NEW LINE (Use this instead):
print("Loading data with FAKE embeddings (Safe Mode)...")

try:
    # 1. Read the CSV
    csv_path = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\restaurant_rag_data.csv"
    df = pd.read_csv(csv_path)
    
    # 2. Limit to 50 rows (Optional for fake embeddings, but good practice)
    df = df.head(50) 
    print(f"Processing {len(df)} restaurants...")

    # 3. Use the FAKE embeddings (Free, runs instantly)
    embeddings = FakeEmbeddings()

    # 4. Create the Vector Store (in memory is fine for testing)
    vectorstore = FAISS.from_texts(
        texts=df['rag_text'].tolist(),
        embedding=embeddings
    )
    
    # 5. Create the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Vector index built successfully (Fake Mode)!")
    
except Exception as e:
    print(f"CRITICAL ERROR loading restaurant data: {e}")
    retriever = None

# Helper function to format retrieved docs
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 3. DEFINE THE RAG CHAIN
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template = """
You are a Restaurant Recommendation Assistant.
Use the User's History to understand their taste, and the Context to find new matches.

USER HISTORY (Past Visits):
{user_history}

AVAILABLE RESTAURANTS (Context):
{context}

USER QUESTION:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Server Error: Restaurant data not loaded.")

    # 1. Look up user history
    user_history = user_profiles.get(request.user_id, "No past history available (New User).")
    
    # 2. Define the chain
    chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "user_history": lambda x: user_history 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 3. Invoke
    try:
        response = chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 instead of 0.0.0.0 for better Windows compatibility
    uvicorn.run(app, host="127.0.0.1", port=8000)