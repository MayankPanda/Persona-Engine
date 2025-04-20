from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from persona_engine import PersonaEngine
from utils import get_session_details

app = FastAPI()
engine = PersonaEngine(provider="openai")  # or "gemini", "sambanova"

class PersonaRequest(BaseModel):
    SessionID: str
    InteractionID: str

@app.post("/generate_persona")
async def generate_persona(payload: PersonaRequest):
    try:
        inputs = get_session_details(payload.SessionID, payload.InteractionID)
        if not inputs:
            raise ValueError("Session details not found")
        result = engine.generate(inputs)
        print(type(result))
        result["Role"]="System"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
