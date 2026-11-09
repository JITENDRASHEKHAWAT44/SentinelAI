from pathlib import Path

from backend.ingestion.loader import ingest_document


# Get project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Specify document path
file_path = BASE_DIR / "data" / "Student_Performance_Analysis_Report.pdf"


def main():

    print("Starting SentinelRAG ingestion test...")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    try:
        # Run ingestion pipeline
        ids = ingest_document(str(file_path))

        print("\nIngestion successful!")
        print(f"Total chunks stored: {len(ids)}")

        print("\nGenerated chunk IDs:")
        for chunk_id in ids:
            print(chunk_id)

    except Exception as error:
        print(f"\nIngestion failed: {error}")


if __name__ == "__main__":
    main()