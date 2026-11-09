import sys
from pathlib import Path
from dotenv import load_dotenv

# Set paths
SECURITY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SECURITY_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(SECURITY_DIR.parent / ".env")

from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path(str(SECURITY_DIR))
rails = LLMRails(config)

response = rails.generate(
    messages=[
        {"role": "user", "content": "Hello there!"}
    ]
)

print(response)
