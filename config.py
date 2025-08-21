# config.py
import os
from dotenv import load_dotenv

# This line loads the .env file for local development.
# On Render, it does nothing, and the code reads the server's environment variables.
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")