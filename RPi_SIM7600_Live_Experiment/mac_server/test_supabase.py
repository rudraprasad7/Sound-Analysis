
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

try:
    result = supabase.table("sound_samples").select("*").limit(1).execute()

    print("SUCCESS: Supabase connected!")
    print("Database response:", result.data)

except Exception as e:
    print("Connection failed:", e)
