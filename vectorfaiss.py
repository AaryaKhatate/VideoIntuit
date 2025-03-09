from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Converts text into a vector embedding."""
    return model.encode([text])[0]

# Ask user to input/paste transcript
transcript = input("Enter your transcript: ")

# Convert transcript to embedding
vector = get_embedding(transcript)

# Initialize FAISS index
vector_dimension = len(vector)
index = faiss.IndexFlatL2(vector_dimension)

# Convert vector to correct shape and add to FAISS
vector = np.array([vector]).astype('float32')
index.add(vector)

print("\nTranscript successfully stored in FAISS vector database!")

# Retrieve the vector to confirm storage
D, I = index.search(vector, 1)  # Searching for the same vector
print("Retrieved vector index:", I[0])
for i in range(index.ntotal):
    print(f"Vector {i}:", index.reconstruct(i))

