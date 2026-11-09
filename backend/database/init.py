from backend.database.vector_store import embeddings

text = "Artificial Intelligence is transforming the world."

result = embeddings.embed_query(text)

print("Embedding generated!")
print("Dimensions:", len(result))
print("First 5 values:", result[:5])