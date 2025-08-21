# config.py
# Import the variables directly from our app_secrets file
import app_secrets

# Supabase and Gemini Keys
SUPABASE_URL = app_secrets.SUPABASE_URL
SUPABASE_SERVICE_KEY = app_secrets.SUPABASE_SERVICE_KEY
GEMINI_API_KEY = app_secrets.GEMINI_API_KEY

# --- NEW LINES ---
# SendGrid Keys
SENDGRID_API_KEY = app_secrets.SENDGRID_API_KEY
SENDGRID_FROM_EMAIL = app_secrets.SENDGRID_FROM_EMAIL