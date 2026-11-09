from backend.retrievel.retriever import retrieve_context


query = "What is supervised learning?"

results = retrieve_context(query, k=3)

print("\nRetrieved Documents:\n")

for index, result in enumerate(results, start=1):

    print(f"\n--- Result {index} ---")

    print("Content:")
    print(result["content"])

    print("Metadata:")
    print(result["metadata"])