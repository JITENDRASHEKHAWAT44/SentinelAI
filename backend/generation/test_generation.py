import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.generation.generation import generate_answer

while True:
    question = input("What's your question ? :")

    answer = generate_answer(question)

    print("\nSentinelRAG Answer:\n")
    print(answer)
