import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(supabase_url=os.getenv("SUPABASE_URL"),supabase_key=os.getenv("SUPABASE_KEY"))