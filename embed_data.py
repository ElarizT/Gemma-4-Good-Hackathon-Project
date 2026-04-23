from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load a tiny, fast math model (only ~100MB)
# This model turns sentences into vectors of 384 numbers.
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup your local "brain" (Database)
client = chromadb.PersistentClient(path="./survival_db")
collection = client.get_or_create_collection(name="emergency_manual")

# 3. Sample Data (Replace this with text from your PDF later)
documents = [
    "To treat a snake bite, keep the limb still and below heart level. Do not cut the wound.",
    "For heatstroke, move the person to a cool place and use wet cloths to lower their temperature.",
    "If someone is choking, perform the Heimlich maneuver behind them."
]

# 4. Turn text into Math and save it
for i, doc in enumerate(documents):
    vector = embed_model.encode(doc).tolist()
    collection.add(
        ids=[f"id_{i}"],
        embeddings=[vector],
        documents=[doc]
    )

print("✅ Survival data successfully turned into math vectors!")