# app.py  (top-level FastAPI app)

from backend.api import app as backend_app
from agents.chat_agent import router as agent_router

# Add your agent routes to the existing backend app
backend_app.include_router(agent_router)

app = backend_app