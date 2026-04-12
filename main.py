# =========================
# 1. IMPORTS
# =========================
from fastapi import FastAPI, Form, UploadFile, File
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import json
import random
import uuid
from PyPDF2 import PdfReader

# =========================
# 2. LOAD ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# =========================
# 3. INIT APP
# =========================
app = FastAPI(
    title="Bias-Free MCQ Assessment System",
    description="AI-powered MCQ generator with fairness and bias correction.",
    version="1.0.0"
)

# Session-based storage (replaces unsafe global list)
stored_sessions: dict[str, list] = {}

# Valid difficulty levels
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


# =========================
# 4. PYDANTIC MODELS
# =========================
class SubmitRequest(BaseModel):
    session_id: str
    answers: list[str]


# =========================
# 5. HELPER FUNCTIONS
# =========================

# 📄 Extract PDF text
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
import re

def safe_json_parse(raw_response: str):
    import re
    import json

    # Remove markdown
    raw_response = raw_response.replace("```json", "").replace("```", "").strip()

    # Extract JSON array
    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found.")

    cleaned = match.group(0)

    # Fix trailing commas
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    # 🔥 VERY IMPORTANT: escape quotes inside values
    cleaned = re.sub(r'(?<=: ")(.*?)(?="[,}])',
                     lambda m: m.group(0).replace('"', '\\"'),
                     cleaned)

    # Remove newlines inside strings
    cleaned = cleaned.replace("\n", " ")

    # Remove control chars
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)

    return json.loads(cleaned)

# 🧠 Generate MCQs using Groq LLM
def generate_mcqs_from_llm(content: str, num_questions: int, difficulty: str) -> list:
    prompt = prompt = f"""
Generate {num_questions} {difficulty} multiple-choice questions based on the following topic:

{content}

Return ONLY a valid JSON array. Do not include any explanation outside JSON.

Format:
[
  {{
    "question": "Question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "A",
    "explanation": "Short explanation"
  }}
]

Rules:
- Each question must have exactly 4 options
- "answer" must be one of: A, B, C, or D
- Do NOT include labels like A), B) inside options
- Keep explanations short and clear
- Avoid special characters and unnecessary symbols
- Ensure valid JSON (no trailing commas, properly closed brackets)
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON API. You output only valid JSON arrays, nothing else."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # lower = more predictable output
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            raw = raw.strip()

            # Extract just the array portion
            start = raw.find('[')
            end   = raw.rfind(']') + 1
            if start == -1 or end == 0:
                print(f"Attempt {attempt+1}: No JSON array found")
                continue

            raw = raw[start:end]

            # Fix common LLM JSON mistakes
            raw = re.sub(r",\s*}", "}", raw)   # trailing comma before }
            raw = re.sub(r",\s*]", "]", raw)   # trailing comma before ]
            raw = raw.replace("\n", " ")
            raw = re.sub(r"[\x00-\x1f\x7f]", "", raw)

            parsed = json.loads(raw)

            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt+1} JSON error: {e}")
            print(f"Raw was: {raw[:300]}")
            continue
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            continue

    return []


# 🔥 Bias Correction — Balance answer distribution across A/B/C/D
def rebalance_answers(mcqs: list) -> list:
    labels = ["A", "B", "C", "D"]
    n = len(mcqs)

    # Create perfectly balanced distribution
    base = n // 4
    remainder = n % 4

    target_labels = []
    for i, label in enumerate(labels):
        count = base + (1 if i < remainder else 0)
        target_labels.extend([label] * count)

    # Shuffle to avoid predictable ordering
    random.shuffle(target_labels)

    for i, q in enumerate(mcqs):
        # Guard: ensure answer index is valid
        answer_index = ord(q["answer"].upper()) - 65
        if answer_index < 0 or answer_index > 3:
            raise ValueError(f"Invalid answer '{q['answer']}' in question {i+1}")

        correct_value = q["options"][answer_index]

        # Shuffle all options randomly
        random.shuffle(q["options"])

        # Determine target position for correct answer
        target_label = target_labels[i]
        target_index = ord(target_label) - 65

        # Remove correct answer from shuffled list and re-insert at target position
        if correct_value in q["options"]:
            q["options"].remove(correct_value)
        q["options"].insert(target_index, correct_value)

        # Update the answer label to match new position
        q["answer"] = target_label

    return mcqs


# 🔥 Normalize Option Lengths — Prevent correct answer from standing out by length
def normalize_options(mcqs: list) -> list:
    for q in mcqs:
        # Simply trim overly long options — do NOT pad with spaces (bad in JSON/UI)
        q["options"] = [opt[:120].strip() for opt in q["options"]]
    return mcqs


# 🙈 Hide answers for exam mode
def hide_answers(mcqs: list) -> list:
    return [
        {
            "question": q["question"],
            "options": q["options"]
        }
        for q in mcqs
    ]


# =========================
# 6. ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "Bias-Free MCQ Assessment System is Running 🚀",
        "endpoints": {
            "generate": "POST /generate-mcq",
            "submit": "POST /submit-answers"
        }
    }


# =========================
# 7. GENERATE MCQ API
# =========================
@app.post("/generate-mcq")
async def generate_mcq(
    topic: str = Form(None),
    text: str = Form(None),
    file: UploadFile = File(None),
    num_questions: int = Form(...),
    difficulty: str = Form(...),
    mode: str = Form("exam")  # "exam" or "practice"
):
    # --- Input Validation ---
    if difficulty.lower() not in VALID_DIFFICULTIES:
        return {"error": f"Invalid difficulty. Choose from: {', '.join(VALID_DIFFICULTIES)}"}

    if not (1 <= num_questions <= 50):
        return {"error": "num_questions must be between 1 and 50."}

    if mode not in {"exam", "practice"}:
        return {"error": "mode must be 'exam' or 'practice'."}

    try:
        content = ""

        # --- Input Source Handling ---
        if topic:
            content = topic.strip()

        elif text:
            content = text.strip()

        elif file:
            file_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
            try:
                with open(file_path, "wb") as f:
                    f.write(await file.read())

                if file.filename.lower().endswith(".pdf"):
                    content = extract_text_from_pdf(file_path)
                else:
                    return {"error": "Unsupported file type. Only PDF is supported."}

            finally:
                # FIX: Always clean up temp file to prevent disk/security leaks
                if os.path.exists(file_path):
                    os.remove(file_path)

        else:
            return {"error": "No input provided. Send a topic, text, or PDF file."}

        if not content.strip():
            return {"error": "Extracted content is empty. Please provide valid input."}

        # Limit content size to avoid LLM token overflow
        content = content[:3000]

        # --- Generate & Process MCQs ---
        mcqs = generate_mcqs_from_llm(content, num_questions, difficulty.lower())
        if not mcqs:
            return {"error": "AI failed to generate valid MCQs. Try again or simplify input."}

        mcqs = [
    q for q in mcqs
    if isinstance(q, dict)
    and "question" in q
    and isinstance(q.get("options"), list)
    and len(q["options"]) == 4
    and q.get("answer") in ["A", "B", "C", "D"]
]

        if not mcqs or not isinstance(mcqs, list):
            return {"error": "Failed to generate MCQs. Please try again."}
        if len(mcqs) < num_questions // 2:
            return {"error": "Too many invalid questions generated. Try again."}

        # Apply bias correction and normalization
        mcqs = rebalance_answers(mcqs)
        mcqs = normalize_options(mcqs)

        # FIX: Use session ID instead of global variable for thread safety
        session_id = str(uuid.uuid4())
        stored_sessions[session_id] = mcqs

        # Return based on mode
        if mode == "exam":
            return {
                "session_id": session_id,
                "mode": "exam",
                "total_questions": len(mcqs),
                "mcqs": hide_answers(mcqs)
            }

        return {
            "session_id": session_id,
            "mode": "practice",
            "total_questions": len(mcqs),
            "mcqs": mcqs
        }

    except ValueError as ve:
        return {"error": f"Validation error: {str(ve)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# =========================
# 8. SUBMIT ANSWERS API
# =========================
@app.post("/submit-answers")
def submit_answers(request: SubmitRequest):
    # FIX: Fetch MCQs using session_id (thread-safe, multi-user safe)
    mcqs = stored_sessions.get(request.session_id)

    if not mcqs:
        return {"error": "Session not found or expired. Please generate MCQs first."}

    if len(request.answers) != len(mcqs):
        return {
            "error": f"Answer count mismatch. Expected {len(mcqs)}, got {len(request.answers)}."
        }

    score = 0
    result = []

    for i, q in enumerate(mcqs):
        correct = q["answer"].upper()
        user = request.answers[i].strip().upper()

        # Validate each answer input
        if user not in ["A", "B", "C", "D"]:
            return {"error": f"Invalid answer '{request.answers[i]}' at question {i+1}. Use A, B, C, or D."}

        is_correct = user == correct
        if is_correct:
            score += 1

        result.append({
            "question_no": i + 1,
            "question": q["question"],
            "your_answer": user,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "No explanation provided.")
        })

    percentage = round((score / len(mcqs)) * 100, 2)

    # Optional: grade label
    if percentage >= 90:
        grade = "Excellent 🏆"
    elif percentage >= 75:
        grade = "Good 👍"
    elif percentage >= 50:
        grade = "Average 📘"
    else:
        grade = "Needs Improvement 📝"

    # Clean up session after submission to free memory
    del stored_sessions[request.session_id]

    return {
        "score": score,
        "total": len(mcqs),
        "percentage": percentage,
        "grade": grade,
        "result": result
    }


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)