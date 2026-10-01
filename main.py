import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in environment variables.")

# Gemini Client Init
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Pocket-Py API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

class QuizRequest(BaseModel):
    topic: str
    num_questions: int

@app.get("/")
def home():
    return {"status": "Pocket-Py Backend Running Successfully!"}

# 1. DOUBT SOLVER CHAT ENDPOINT
@app.post("/api/chat")
def chat_with_py_teacher(req: ChatRequest):
    try:
        system_instruction = (
            "You are 'Pocket-Py', an elite, friendly, and practical personal Python teacher. "
            "You cover Python from fundamentals, OOP, memory management, internals, to FAANG/Placement-level "
            "Data Structures, Algorithms, and interview problem-solving tricks. "
            "Keep answers concise, intuitive, code-rich, and easy to grasp. Provide clear syntax and best practices."
        )

        # Context build-up
        contents = []
        for item in req.history:
            contents.append(item)
        contents.append(req.prompt)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. DYNAMIC QUIZ GENERATOR ENDPOINT
@app.post("/api/quiz")
def generate_quiz(req: QuizRequest):
    try:
        prompt = f"""
        Generate a {req.num_questions}-question multiple-choice quiz on the Python topic: '{req.topic}'.
        Difficulty level: practical coding/interview standard.
        
        Strictly output valid JSON matching this schema list:
        [
          {{
            "question": "question text with code snippets if relevant",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_index": 0,
            "explanation": "Short 1-line reason why this is correct"
          }}
        ]
        Do not wrap with markdown code fences (no ```json). Output raw JSON array only.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        data = json.loads(response.text)
        return {"questions": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))