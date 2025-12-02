# backend/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "ai-lab")
DB_PATH = os.getenv("DB_PATH", "manifests/manifests.db")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ai-lab")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")  #3 for later
