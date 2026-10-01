import os
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Pocket-Py API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Priority list: pehle ultra-fast lite model, fir standard flash
MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

class QuizRequest(BaseModel):
    topic: str
    num_questions: int

def call_gemini_with_fallback(contents, system_instruction=None, json_mode=False):
    """503 high-demand error se bachne ke liye auto retry aur fallback mechanism"""
    last_err = None
    for model_name in MODELS:
        for attempt in range(2):
            try:
                config_args = {}
                if system_instruction:
                    config_args["system_instruction"] = system_instruction
                if json_mode:
                    config_args["response_mime_type"] = "application/json"

                cfg = types.GenerateContentConfig(**config_args)
                res = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=cfg
                )
                if res and res.text:
                    return res.text
            except Exception as e:
                last_err = e
                print(f"Attempt failed on {model_name}: {e}. Retrying...")
                time.sleep(1)
    raise last_err

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "Pocket-Py Backend Running Successfully!"}

# 1. DOUBT SOLVER CHAT
@app.post("/api/chat")
def chat_with_py_teacher(req: ChatRequest):
    try:
        system_instruction = (
            "You are 'Pocket-Py', an elite, friendly, and practical personal Python teacher. "
            "You cover Python from fundamentals, OOP, memory management, internals, to FAANG/Placement-level "
            "Data Structures, Algorithms, and interview problem-solving tricks. "
            "Keep answers concise, intuitive, code-rich, and easy to grasp. Provide clear syntax and best practices."
        )

        contents = []
        for item in req.history:
            contents.append(item)
        contents.append(req.prompt)

        text = call_gemini_with_fallback(contents, system_instruction=system_instruction, json_mode=False)
        return {"response": text}
    except Exception as e:
        print(f"Chat Final Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 2. DYNAMIC QUIZ GENERATOR
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
        Do not wrap with markdown code fences. Output raw JSON array only.
        """

        raw_text = call_gemini_with_fallback(prompt, json_mode=True)
        # Markdown cleanup if returned
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        data = json.loads(cleaned.strip())
        return {"questions": data}
    except Exception as e:
        print(f"Quiz Final Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))