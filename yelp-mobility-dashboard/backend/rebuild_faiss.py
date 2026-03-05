import os
import re
import shutil
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------
# Config (edit these paths)
# -------------------------
CSV_PATH = r"C:\Users\lebro\OneDrive - Nanyang Technological University\Github\fyp-demo\yelp-mobility-dashboard\public\data\restaurant_rag_data.csv"
INDEX_FOLDER = "faiss_index_store_local"   # keep same as server.py
DELETE_OLD_INDEX = True                    # set False if you want to keep backup
BATCH_SIZE = 1000


CITY_STATE_RE = re.compile(r"City:\s*([^,]+),\s*([A-Z]{2})\.")

def extract_city_state(rag_text: str):
    m = CITY_STATE_RE.search(str(rag_text or ""))
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    if "rag_text" not in df.columns or "business_id" not in df.columns or "name" not in df.columns:
        raise ValueError("CSV must contain columns: business_id, name, rag_text")

    print(f"Rows: {len(df)}")

    # Parse city/state from rag_text
    print("Extracting city/state from rag_text...")
    df["city"], df["state"] = zip(*df["rag_text"].fillna("").map(extract_city_state))

    # Build docs with metadata
    print("Building documents...")
    docs = []
    skipped = 0

    for _, r in df.iterrows():
        rag_text = r.get("rag_text")
        if not isinstance(rag_text, str) or not rag_text.strip():
            skipped += 1
            continue

        docs.append(
            Document(
                page_content=rag_text,
                metadata={
                    "business_id": str(r.get("business_id", "")),
                    "name": str(r.get("name", "")),
                    "city": (r.get("city") or ""),
                    "state": (r.get("state") or ""),
                }
            )
        )

    print(f"Docs: {len(docs)} (skipped {skipped})")

    # Embeddings (must match server.py)
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Delete old index folder
    if DELETE_OLD_INDEX and os.path.exists(INDEX_FOLDER):
        print(f"Deleting old index folder: {INDEX_FOLDER}")
        shutil.rmtree(INDEX_FOLDER)

    # Build FAISS index in batches (safer for large datasets)
    print("Building FAISS index...")
    if not docs:
        raise ValueError("No documents to index.")

    vectorstore = FAISS.from_documents(docs[:BATCH_SIZE], embedding=embeddings)
    for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[i:i+BATCH_SIZE])
        print(f"Indexed {min(i+BATCH_SIZE, len(docs))}/{len(docs)}")

    print("Saving index...")
    vectorstore.save_local(INDEX_FOLDER)
    print("✅ Done. Saved to:", INDEX_FOLDER)


if __name__ == "__main__":
    main()