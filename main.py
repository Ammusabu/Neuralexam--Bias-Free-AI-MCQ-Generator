# =========================
# 1. IMPORTS (all at top)
# =========================
import re
import os
import json
import random
import uuid
import logging
import time
from contextlib import asynccontextmanager

import pdfplumber
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# =========================
# 2. LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# 3. LOAD ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# =========================
# 4. CONSTANTS
# =========================
VALID_DIFFICULTIES   = {"easy", "medium", "hard"}
VALID_MODES          = {"exam", "practice"}
MAX_QUESTIONS        = 50
MIN_QUESTIONS        = 1
CHUNK_SIZE           = 4000        # chars per chunk (≈600 tokens — enough context)
MAX_CHUNKS           = 5           # max Groq API calls per request
SESSION_TTL_SECONDS  = 3600        # 1 hour session expiry
TEMP_DIR             = "/tmp"      # safe temp directory

# =========================
# 5. SESSION STORE
# =========================
# Each entry: { "mcqs": [...], "created_at": float }
stored_sessions: dict[str, dict] = {}


def purge_expired_sessions() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, data in stored_sessions.items()
        if now - data["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del stored_sessions[sid]
        logger.info(f"Purged expired session: {sid}")


# =========================
# 6. LIFESPAN (replaces deprecated @app.on_event)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NeuralExam MCQ API starting up.")
    yield
    logger.info("NeuralExam MCQ API shutting down.")


# =========================
# 7. INIT APP  (CORS added BEFORE routes)
# =========================
app = FastAPI(
    title="NeuralExam — Bias-Free MCQ Assessment System",
    description="AI-powered MCQ generator with fairness and bias correction.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS must be added before any route registration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 8. PYDANTIC MODELS
# =========================
class SubmitRequest(BaseModel):
    session_id: str
    answers: list[str]


# =========================
# 9. HELPER — PDF EXTRACTION
# =========================
def extract_text_from_pdf(file_path: str) -> str:
    """Extract and clean text from a PDF file."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    raw = " ".join(text_parts)
    return re.sub(r"\s+", " ", raw).strip()


# =========================
# 10. HELPER — SAFE JSON PARSE
# =========================
def safe_json_parse(raw: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Handles markdown fences, trailing commas, stray control chars.
    """
    # Strip markdown fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Find outermost JSON array
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in LLM response.")

    raw = raw[start:end]

    # Fix trailing commas before } or ]
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)

    # Remove control characters (except tab/newline used in JSON)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)

    # Collapse newlines inside string values to spaces
    raw = re.sub(r'(?<=")(\n)(?=[^"]*")', " ", raw)

    return json.loads(raw)


# =========================
# 11. HELPER — NORMALIZE ANSWER KEY
# =========================
def normalize_answer_key(raw_answer: str) -> str:
    """
    Convert LLM answer variants like 'A.', 'Option A', '(A)', 'a' → 'A'.
    Raises ValueError if no valid A–D key found.
    """
    if not raw_answer:
        raise ValueError("Empty answer key.")
    match = re.search(r"\b([A-Da-d])\b", raw_answer)
    if not match:
        raise ValueError(f"Cannot parse answer key from: '{raw_answer}'")
    return match.group(1).upper()


# =========================
# 12. HELPER — GENERATE MCQs FROM LLM
# =========================
def generate_mcqs_from_llm(content: str, num_questions: int, difficulty: str) -> list:
    """
    Call Groq LLM to generate MCQs.  Returns a list of valid question dicts.
    Retries up to 3 times on parse failure.
    """
    prompt = f"""
Generate exactly {num_questions} {difficulty}-level multiple-choice questions from the content below.

CONTENT:
{content}

REQUIREMENTS:
- No repetitive or definition-only questions
- Mix of: conceptual, application, analytical, scenario-based questions
- Each question must test understanding, not rote memorization
- 4 distinct answer choices per question
- All choices similar in length (avoid obvious length cues)
- Distractors must be plausible — not obviously wrong
- Correct answer must NOT follow a pattern (not always A)
- Provide a clear explanation for the correct answer

OUTPUT FORMAT — respond with ONLY a valid JSON array, no markdown, no preamble:
[
  {{
    "question": "Question text here?",
    "options": ["Option text A", "Option text B", "Option text C", "Option text D"],
    "answer": "B",
    "explanation": "Explanation why B is correct."
  }}
]
"""

    for attempt in range(1, 4):
        try:
            logger.info(f"LLM call attempt {attempt} for {num_questions} {difficulty} questions.")
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON API. Output ONLY a valid JSON array "
                            "with no markdown, no explanation, no preamble."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()
            parsed = safe_json_parse(raw)

            if not isinstance(parsed, list) or len(parsed) == 0:
                logger.warning(f"Attempt {attempt}: parsed result is empty or not a list.")
                continue

            # Validate and normalise each question
            valid = []
            for q in parsed:
                try:
                    if not isinstance(q, dict):
                        continue
                    if not q.get("question") or not isinstance(q.get("options"), list):
                        continue
                    if len(q["options"]) != 4:
                        continue
                    q["answer"] = normalize_answer_key(str(q.get("answer", "")))
                    valid.append(q)
                except ValueError as ve:
                    logger.warning(f"Skipping question — {ve}")
                    continue

            if valid:
                logger.info(f"Attempt {attempt}: got {len(valid)} valid questions.")
                return valid

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt} JSON error: {e}")
        except Exception as e:
            logger.error(f"Attempt {attempt} unexpected error: {e}")

    return []


# =========================
# 13. HELPER — SPLIT TEXT INTO CHUNKS
# =========================
def split_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping chunks to preserve context across boundaries."""
    chunks = []
    overlap = 200  # char overlap to preserve context at boundaries
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return chunks


# =========================
# 14. HELPER — BIAS CORRECTION
# =========================
def rebalance_answers(mcqs: list) -> list:
    """
    Redistribute correct answers so A/B/C/D each appear ~equally.
    Shuffles options array and updates the answer key accordingly.
    """
    labels = ["A", "B", "C", "D"]
    n = len(mcqs)

    # Build balanced target distribution
    base      = n // 4
    remainder = n % 4
    target_labels = []
    for i, label in enumerate(labels):
        count = base + (1 if i < remainder else 0)
        target_labels.extend([label] * count)
    random.shuffle(target_labels)

    rebalanced = []
    for i, q in enumerate(mcqs):
        try:
            answer_index = ord(q["answer"]) - 65          # 'A'→0 … 'D'→3
            if not (0 <= answer_index <= 3):
                logger.warning(f"Q{i+1}: invalid answer index, skipping rebalance.")
                rebalanced.append(q)
                continue

            correct_text = q["options"][answer_index]

            # Shuffle all options
            opts = q["options"][:]
            random.shuffle(opts)

            # Place correct answer at the target position
            target_label = target_labels[i]
            target_idx   = ord(target_label) - 65

            if correct_text in opts:
                opts.remove(correct_text)
            opts.insert(target_idx, correct_text)

            q["options"] = opts
            q["answer"]  = target_label
            rebalanced.append(q)

        except Exception as e:
            logger.warning(f"Q{i+1} rebalance error: {e} — keeping original.")
            rebalanced.append(q)

    return rebalanced


# =========================
# 15. HELPER — NORMALIZE OPTION LENGTHS
# =========================
def normalize_options(mcqs: list) -> list:
    """Trim excessively long options to prevent length-based answer leaking."""
    for q in mcqs:
        q["options"] = [opt.strip()[:150] for opt in q["options"]]
    return mcqs


# =========================
# 16. HELPER — HIDE ANSWERS (exam mode)
# =========================
def hide_answers(mcqs: list) -> list:
    return [
        {"question": q["question"], "options": q["options"]}
        for q in mcqs
    ]


# =========================
# 17. ROUTES
# =========================

@app.get("/")
def home():
    return {
        "message": "NeuralExam Bias-Free MCQ API is running 🚀",
        "version": "2.0.0",
        "endpoints": {
            "generate": "POST /generate-mcq",
            "submit":   "POST /submit-answers",
            "health":   "GET  /health",
        },
    }


@app.get("/health")
def health():
    purge_expired_sessions()
    return {
        "status": "ok",
        "active_sessions": len(stored_sessions),
    }


# =========================
# 18. GENERATE MCQ ENDPOINT
# =========================
@app.post("/generate-mcq")
async def generate_mcq(
    topic:         str        = Form(None),
    text:          str        = Form(None),
    file:          UploadFile = File(None),
    num_questions: int        = Form(...),
    difficulty:    str        = Form(...),
    mode:          str        = Form("exam"),
):
    purge_expired_sessions()

    # --- Validate inputs ---
    difficulty = difficulty.strip().lower()
    mode       = mode.strip().lower()

    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(400, f"Invalid difficulty. Choose from: {', '.join(VALID_DIFFICULTIES)}")

    if not (MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS):
        raise HTTPException(400, f"num_questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}.")

    if mode not in VALID_MODES:
        raise HTTPException(400, "mode must be 'exam' or 'practice'.")

    # --- Build content string ---
    content = ""

    if topic and topic.strip():
        content = topic.strip()

    elif text and text.strip():
        content = text.strip()

    elif file:
        # Sanitize filename — prevent path traversal
        safe_name = re.sub(r"[^\w.\-]", "_", os.path.basename(file.filename or "upload"))
        file_path = os.path.join(TEMP_DIR, f"mcq_{uuid.uuid4().hex}_{safe_name}")

        try:
            file_bytes = await file.read()
            if len(file_bytes) > 10 * 1024 * 1024:   # 10 MB limit
                raise HTTPException(400, "File too large. Maximum size is 10 MB.")

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            if not safe_name.lower().endswith(".pdf"):
                raise HTTPException(400, "Unsupported file type. Only PDF is supported.")

            content = extract_text_from_pdf(file_path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File processing error: {e}")
            raise HTTPException(500, f"Failed to process file: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        raise HTTPException(400, "No input provided. Send a topic, text, or PDF file.")

    if not content.strip():
        raise HTTPException(400, "Extracted content is empty. Please provide valid input.")

    # --- Chunk content and generate MCQs ---
    chunks   = split_text(content, CHUNK_SIZE)[:MAX_CHUNKS]
    n_chunks = len(chunks)

    # Distribute questions evenly across chunks, ensuring total >= num_questions
    base_per_chunk = max(3, (num_questions + n_chunks - 1) // n_chunks)

    all_mcqs: list = []
    for idx, chunk in enumerate(chunks):
        # Request slightly more per chunk to compensate for filtered-out bad questions
        request_count = min(base_per_chunk + 2, 15)
        logger.info(f"Chunk {idx+1}/{n_chunks}: requesting {request_count} questions.")
        part = generate_mcqs_from_llm(chunk, request_count, difficulty)
        if part:
            all_mcqs.extend(part)
        # Stop early if we already have enough
        if len(all_mcqs) >= num_questions * 1.5:
            break

    # De-duplicate by question text (case-insensitive)
    seen: set[str] = set()
    unique_mcqs: list = []
    for q in all_mcqs:
        key = q["question"].strip().lower()[:80]
        if key not in seen:
            seen.add(key)
            unique_mcqs.append(q)

    mcqs = unique_mcqs[:num_questions]

    # Validate minimum yield
    if len(mcqs) < max(1, num_questions // 2):
        raise HTTPException(
            500,
            f"AI only generated {len(mcqs)} valid questions (requested {num_questions}). "
            "Try a longer text or a different topic."
        )

    # --- Apply bias corrections ---
    mcqs = rebalance_answers(mcqs)
    mcqs = normalize_options(mcqs)

    # --- Store session ---
    session_id = str(uuid.uuid4())
    stored_sessions[session_id] = {
        "mcqs":       mcqs,
        "created_at": time.time(),
        "mode":       mode,
    }
    logger.info(f"Session {session_id} created with {len(mcqs)} questions.")

    # --- Return response ---
    response_mcqs = hide_answers(mcqs) if mode == "exam" else mcqs

    return {
        "session_id":      session_id,
        "mode":            mode,
        "total_questions": len(mcqs),
        "mcqs":            response_mcqs,
    }


# =========================
# 19. SUBMIT ANSWERS ENDPOINT
# =========================
@app.post("/submit-answers")
def submit_answers(request: SubmitRequest):
    purge_expired_sessions()

    session = stored_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired. Please generate MCQs first.")

    mcqs = session["mcqs"]

    if len(request.answers) != len(mcqs):
        raise HTTPException(
            400,
            f"Answer count mismatch. Expected {len(mcqs)}, got {len(request.answers)}."
        )

    score  = 0
    result = []

    for i, q in enumerate(mcqs):
        raw_user = request.answers[i].strip().upper()

        # Validate answer format
        if raw_user not in ["A", "B", "C", "D"]:
            raise HTTPException(
                400,
                f"Invalid answer '{request.answers[i]}' at question {i+1}. Use A, B, C, or D."
            )

        correct    = q["answer"].upper()
        is_correct = raw_user == correct
        if is_correct:
            score += 1

        result.append({
            "question_no":     i + 1,
            "question":        q["question"],
            "your_answer":     raw_user,
            "correct_answer":  correct,
            "is_correct":      is_correct,
            "explanation":     q.get("explanation", "No explanation provided."),
        })

    total      = len(mcqs)
    percentage = round((score / total) * 100, 2)

    if   percentage >= 90: grade = "Excellent 🏆"
    elif percentage >= 75: grade = "Good 👍"
    elif percentage >= 50: grade = "Average 📘"
    else:                  grade = "Needs Improvement 📝"

    # Clean up session after grading
    del stored_sessions[request.session_id]
    logger.info(f"Session {request.session_id} graded and removed. Score: {score}/{total}")

    return {
        "score":      score,
        "total":      total,
        "percentage": percentage,
        "grade":      grade,
        "result":     result,
    }