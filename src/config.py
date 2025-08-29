# config.py
# Central configuration: API key, OpenAI client factory, constants.
# pip install --upgrade google-genai

from openai import OpenAI
from google import genai

# --- HARD-CODED API KEY 
OPENAI_API_KEY = ""

GEMINI_API_KEY = ""

# --- Embedding constants ---
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# --- Chat model for extraction ---
IE_MODEL = "gpt-4o-mini"

def openai_client() -> OpenAI:
    """Factory for an OpenAI client using the single configured key."""
    return OpenAI(api_key=OPENAI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"
def gemini_client() -> genai:
    """Factory for a Gemini client using the single configured key."""
    return  genai.Client(api_key=GEMINI_API_KEY)
