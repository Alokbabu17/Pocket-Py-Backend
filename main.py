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


# ============================================================
# GEMINI MODEL PRIORITY
# ============================================================

MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash"
]


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    prompt: str
    history: list = []


class QuizRequest(BaseModel):
    topic: str
    num_questions: int


# ============================================================
# GEMINI FALLBACK SYSTEM
# ============================================================

def call_gemini_with_fallback(
    contents,
    system_instruction=None,
    json_mode=False
):
    """
    503 high-demand error se bachne ke liye
    automatic retry aur fallback mechanism.
    """

    last_err = None

    for model_name in MODELS:

        for attempt in range(2):

            try:

                config_args = {}

                if system_instruction:
                    config_args["system_instruction"] = (
                        system_instruction
                    )

                if json_mode:
                    config_args[
                        "response_mime_type"
                    ] = "application/json"

                cfg = types.GenerateContentConfig(
                    **config_args
                )

                res = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=cfg
                )

                if res and res.text:
                    return res.text

            except Exception as e:

                last_err = e

                print(
                    f"Attempt failed on "
                    f"{model_name}: {e}. Retrying..."
                )

                time.sleep(1)

    raise last_err


# ============================================================
# HOME
# ============================================================

@app.api_route("/", methods=["GET", "HEAD"])
def home():

    return {
        "status":
        "Pocket-Py Backend Running Successfully!"
    }


# ============================================================
# 1. DOUBT SOLVER CHAT
# ============================================================

@app.post("/api/chat")
def chat_with_py_teacher(req: ChatRequest):

    try:

        system_instruction = (
            "You are 'Pocket-Py', an elite, friendly, "
            "and practical personal Python teacher. "

            "You cover Python from fundamentals, "
            "OOP, memory management, internals, "
            "to FAANG/Placement-level Data Structures, "
            "Algorithms, and interview problem-solving tricks. "

            "Keep answers concise, intuitive, "
            "code-rich, and easy to grasp. "

            "Provide clear syntax and best practices."
        )

        contents = []

        for item in req.history:
            contents.append(item)

        contents.append(req.prompt)

        text = call_gemini_with_fallback(
            contents,
            system_instruction=system_instruction,
            json_mode=False
        )

        return {
            "response": text
        }

    except Exception as e:

        print(
            f"Chat Final Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 2. DYNAMIC QUIZ GENERATOR
# ============================================================

@app.post("/api/quiz")
def generate_quiz(req: QuizRequest):

    try:

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if req.num_questions < 1:
            raise HTTPException(
                status_code=400,
                detail="Number of questions must be at least 1."
            )

        if req.num_questions > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 questions are allowed."
            )

        topic = req.topic.strip()

        if not topic:
            raise HTTPException(
                status_code=400,
                detail="Topic cannot be empty."
            )

        # ----------------------------------------------------
        # Detect special quiz types
        # ----------------------------------------------------

        is_leetcode = (
            "LeetCode / Word Problems" in topic
        )

        is_faang = (
            "FAANG / Placement Questions" in topic
        )

        is_dsa = (
            topic.startswith("DSA")
        )

        # ----------------------------------------------------
        # Base quiz instruction
        # ----------------------------------------------------

        base_instruction = """
You are Pocket-Py, an expert Python instructor,
competitive programming mentor, DSA teacher,
and technical interview preparation mentor.

You are generating questions for a personalized
Python learning application.

The selected topic is provided as a hierarchy:

Category > Subcategory > Specific Topic

You MUST generate questions specifically about the
MOST SPECIFIC topic at the end of the hierarchy.

Do NOT randomly mix unrelated Python topics.

For example:

If the topic is:
Collections > List > append()

focus mainly on append().

If the topic is:
OOP — Classes & Objects > Inheritance > MRO

focus specifically on Method Resolution Order.

If the topic is:
DSA > Graphs > Dijkstra

focus specifically on Dijkstra's algorithm.

Questions may contain Python code when useful.

Difficulty should be practical and interview-oriented,
but remain appropriate for the selected topic.

Avoid duplicate questions.

Each question must have exactly four options.

There must be exactly one correct answer.

The correct_index must be zero-based:

0 = first option
1 = second option
2 = third option
3 = fourth option.

The explanation should be concise.

Return ONLY a valid JSON array.
Do not use Markdown.
Do not add any text outside the JSON.
"""

        # ----------------------------------------------------
        # LEETCODE / WORD PROBLEM MODE
        # ----------------------------------------------------

        if is_leetcode:

            special_instruction = """
SPECIAL MODE: LEETCODE / WORD PROBLEMS

This is NOT a normal short conceptual MCQ.

Generate a proper programming problem.

The question should feel like a real
LeetCode-style problem.

Include when appropriate:

- A meaningful problem scenario
- Clear input description
- Expected output
- Constraints
- Example input/output
- Important edge cases
- Algorithmic reasoning

The problem must be solvable from the information
contained inside the question itself.

The four options should represent possible
solutions, approaches, outputs, algorithms,
or code snippets depending on the problem.

Do NOT make the question merely:

"What is a list?"

or:

"Which method is used?"

Instead create a genuine problem-solving question.

For example, if the selected topic is:

LeetCode / Word Problems > Arrays > Medium Array Problems

create a medium-level array problem.

If it is:

LeetCode / Word Problems > Dynamic Programming > Hard DP Problems

create a difficult dynamic-programming problem.

Difficulty MUST respect the selected difficulty level
when Easy, Medium, or Hard is present.

The learner should need to reason about the problem
to select the correct option.
"""

        # ----------------------------------------------------
        # FAANG MODE
        # ----------------------------------------------------

        elif is_faang:

            special_instruction = """
SPECIAL MODE: FAANG / PLACEMENT

Generate placement and technical interview-level
questions.

Questions should test:

- Code understanding
- Output prediction
- Complexity
- Debugging
- Python concepts
- DSA reasoning
- Optimal approaches
- Interview scenarios

The question should be more challenging than a
basic textbook definition.

If the selected topic is specifically a Python topic,
stay within that topic.

If the selected topic is DSA-related, use an
appropriate coding/interview problem.

Use realistic interview-style distractors.
"""

        # ----------------------------------------------------
        # DSA MODE
        # ----------------------------------------------------

        elif is_dsa:

            special_instruction = """
SPECIAL MODE: DATA STRUCTURES AND ALGORITHMS

Generate DSA-oriented questions.

Focus on:

- Algorithmic reasoning
- Time complexity
- Space complexity
- Data structure behavior
- Code tracing
- Correct algorithm selection
- Edge cases
- Python implementations when useful

Do NOT generate generic Python syntax questions
unless the selected topic specifically requires them.

For example:

DSA > Arrays > Sliding Window

must focus on sliding-window reasoning.

DSA > Trees > Inorder Traversal

must focus on inorder traversal.

DSA > Dynamic Programming > Knapsack

must focus on knapsack/DP reasoning.
"""

        # ----------------------------------------------------
        # NORMAL PYTHON MODE
        # ----------------------------------------------------

        else:

            special_instruction = """
SPECIAL MODE: PYTHON TOPIC LEARNING

Generate focused Python questions based on the
specific selected topic.

Questions can include:

- Conceptual understanding
- Output prediction
- Code tracing
- Debugging
- Practical usage
- Common mistakes
- Interview-style reasoning

Do not unnecessarily introduce unrelated concepts.
"""

        # ----------------------------------------------------
        # FINAL PROMPT
        # ----------------------------------------------------

        prompt = f"""
{base_instruction}

{special_instruction}

SELECTED TOPIC:
{topic}

NUMBER OF QUESTIONS:
{req.num_questions}

OUTPUT SCHEMA:

[
  {{
    "question": "question text with code snippets if relevant",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "correct_index": 0,
    "explanation": "Short reason why the correct answer is correct."
  }}
]

IMPORTANT VALIDATION RULES:

1. Return exactly {req.num_questions} questions.

2. Return exactly four options for every question.

3. correct_index must be 0, 1, 2, or 3.

4. There must be exactly one correct option.

5. The correct answer must actually solve the
   displayed problem.

6. Do not create ambiguous questions.

7. Do not repeat the same question.

8. Do not reveal the correct answer in the question.

9. For code questions, carefully trace the code
   before selecting the correct answer.

10. For numerical or complexity questions,
    verify the answer before returning it.

11. For LeetCode-style questions, make the problem
    sufficiently detailed and reasoning-oriented.

12. Keep explanations short but technically accurate.

13. Return raw JSON only.
"""

        # ----------------------------------------------------
        # CALL GEMINI
        # ----------------------------------------------------

        raw_text = call_gemini_with_fallback(
            prompt,
            json_mode=True
        )

        # ----------------------------------------------------
        # CLEAN MARKDOWN IF GEMINI RETURNS CODE FENCE
        # ----------------------------------------------------

        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        if cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        data = json.loads(cleaned)

        # ----------------------------------------------------
        # BASIC RESPONSE VALIDATION
        # ----------------------------------------------------

        if not isinstance(data, list):

            raise ValueError(
                "Gemini returned invalid quiz format."
            )

        if len(data) != req.num_questions:

            raise ValueError(
                "Gemini returned an incorrect number "
                "of questions."
            )

        for question in data:

            if not isinstance(question, dict):
                raise ValueError(
                    "Invalid question object."
                )

            if "question" not in question:
                raise ValueError(
                    "Question text missing."
                )

            if "options" not in question:
                raise ValueError(
                    "Options missing."
                )

            if "correct_index" not in question:
                raise ValueError(
                    "Correct index missing."
                )

            if "explanation" not in question:
                raise ValueError(
                    "Explanation missing."
                )

            if len(question["options"]) != 4:
                raise ValueError(
                    "Every question must have exactly "
                    "four options."
                )

            correct_index = question["correct_index"]

            if correct_index not in [0, 1, 2, 3]:
                raise ValueError(
                    "Invalid correct_index."
                )

        # ----------------------------------------------------
        # RETURN TO FLUTTER
        # ----------------------------------------------------

        return {
            "questions": data
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Quiz Final Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )