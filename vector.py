from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Converts text into a normalized vector embedding."""
    vector = model.encode([text])[0]
    return vector / np.linalg.norm(vector)  # Normalize for cosine similarity

# Initialize FAISS index
vector_dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(vector_dimension)  # Inner Product (cosine similarity after normalization)
transcripts = []  # Store transcripts for reference

while True:
    transcript = input("Enter transcript (or 'exit' to stop): ")
    if transcript.lower() == "exit":
        break

    vector = get_embedding(transcript)
    vector = np.array([vector]).astype('float32')

    index.add(vector)  # Add to FAISS index
    transcripts.append(transcript)  # Store transcript text

print("\nAll transcripts successfully stored in FAISS vector database!")

# Search for similar transcripts
query = input("\nEnter a transcript to find similar ones: ")
query_vector = get_embedding(query)
query_vector = np.array([query_vector]).astype('float32')

k = min(3, len(transcripts))  # Retrieve top 3 similar transcripts
D, I = index.search(query_vector, k)  # Search for top-k similar vectors

print("\nMost similar transcripts:")
for idx, score in zip(I[0], D[0]):
    print(f"Similarity Score: {score:.4f} | Transcript: {transcripts[idx]}")
