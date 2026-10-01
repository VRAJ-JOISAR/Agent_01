from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from google import genai


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================

knowledge_folder = "knowledge"

documents = []

for filename in os.listdir(knowledge_folder):

    filepath = os.path.join(
        knowledge_folder,
        filename
    )

    # Ignore directories
    if not os.path.isfile(filepath):
        continue

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    documents.append({
        "filename": filename,
        "text": text
    })


print("Documents loaded:", len(documents))


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

chunks = []

for document in documents:

    # For our simple project,
    # each paragraph becomes one chunk

    paragraphs = document["text"].split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

            chunks.append({
                "text": paragraph,
                "source": document["filename"]
            })


print("Number of chunks:", len(chunks))


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE EMBEDDINGS FOR DOCUMENT CHUNKS
# ============================================================

print("Creating embeddings...")

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)

embeddings = np.array(
    embeddings
).astype("float32")


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# 5. CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)


print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# ============================================================
# 6. GET USER QUESTION
# ============================================================

query = input(
    "\nAsk a motor question: "
)


# ============================================================
# 7. CONVERT QUESTION INTO EMBEDDING
# ============================================================

query_embedding = embedding_model.encode(
    [query]
)

query_embedding = np.array(
    query_embedding
).astype("float32")


# ============================================================
# 8. SIMILARITY SEARCH
# ============================================================

k = 3

distances, indices = index.search(
    query_embedding,
    k
)


# ============================================================
# 9. RETRIEVE TOP-K CHUNKS
# ============================================================

retrieved_chunks = []

for i, idx in enumerate(indices[0]):

    retrieved_chunks.append({
        "text": chunks[idx]["text"],
        "source": chunks[idx]["source"],
        "distance": distances[0][i]
    })


# ============================================================
# 10. DISPLAY RETRIEVED CHUNKS
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVED CONTEXT")
print("=" * 70)

for i, chunk in enumerate(
    retrieved_chunks
):

    print(
        f"\n--- Chunk {i + 1} ---"
    )

    print(
        "Source:",
        chunk["source"]
    )

    print(
        "Distance:",
        chunk["distance"]
    )

    print(
        "Text:",
        chunk["text"]
    )


# ============================================================
# 11. CREATE CONTEXT FOR LLM
# ============================================================

context_parts = []

for chunk in retrieved_chunks:

    context_parts.append(
        f"Source: {chunk['source']}\n"
        f"{chunk['text']}"
    )


context = "\n\n".join(
    context_parts
)


# ============================================================
# 12. CREATE RAG PROMPT
# ============================================================

prompt = f"""
You are a motor maintenance assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the
context, clearly say:

"The information is not available
in the provided knowledge base."

Do not invent technical information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

User Question:

{query}

Answer:
"""


# ============================================================
# 13. CONNECT TO GEMINI
# ============================================================

client = genai.Client()


# ============================================================
# 14. SEND PROMPT TO GEMINI
# ============================================================

print("\n")
print("=" * 70)
print("SENDING CONTEXT TO GEMINI...")
print("=" * 70)


response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)


# ============================================================
# 15. GET GEMINI'S ANSWER
# ============================================================

answer = response.text


# ============================================================
# 16. DISPLAY FINAL ANSWER
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(answer)