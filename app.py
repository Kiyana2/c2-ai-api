
# app.py

import os
import uuid
import threading

import fitz
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

from conversation import (
    create_conversation,
    get_conversation,
    add_message,
    clear_conversation,
    delete_conversation,
)

# ============================================================
# API SETUP
# ============================================================

app = FastAPI(title="CATCH AI Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "qwen/qwen3.8-27b"

# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    print(f"Opening {pdf_path}...")

    chunks = []

    with fitz.open(pdf_path) as doc:
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()

            if len(text) < 50:
                try:
                    textpage = page.get_textpage_ocr(
                        flags=3,
                        language="eng"
                    )
                    text = page.get_text("text", textpage=textpage).strip()
                except Exception as exc:
                    print(
                        f"OCR failed on page {page_num + 1}: {exc}"
                    )
                    text = ""

            full_text.append(text)

    words = " ".join(full_text).split()

    word_chunk_size = max(1, chunk_size // 5)
    word_overlap = max(0, overlap // 5)
    step = max(1, word_chunk_size - word_overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + word_chunk_size])

        if len(chunk.strip()) > 10:
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks from {pdf_path}")
    return chunks


# ============================================================
# RETRIEVAL DATABASE
# ============================================================

class SimpleVectorDB:
    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def add_documents(self, text_chunks):
        if not text_chunks:
            return

        vectors = vectorizer.transform(text_chunks).toarray()

        self.chunks.extend(text_chunks)

        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack(
                [self.embeddings, vectors]
            )

    def search(self, query, top_k=3):
        if not self.chunks or self.embeddings is None:
            return []

        query_vector = vectorizer.transform([query]).toarray()[0]

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return self.chunks[:top_k]

        document_norms = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        denominator = document_norms * query_norm

        similarities = np.divide(
            self.embeddings @ query_vector,
            denominator,
            out=np.zeros_like(document_norms),
            where=denominator != 0
        )

        indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[i] for i in indices]


pdf_files = [
    "catch_framwork.pdf",
    "scenarios.pdf",
]

all_text_chunks = []

for pdf_file in pdf_files:
    if not os.path.exists(pdf_file):
        raise RuntimeError(f"Required PDF not found: {pdf_file}")

    all_text_chunks.extend(
        extract_and_chunk_pdf(pdf_file)
    )

if not all_text_chunks:
    raise RuntimeError("No text could be extracted from the PDF files.")

vectorizer = TfidfVectorizer()
vectorizer.fit(all_text_chunks)

vector_db = SimpleVectorDB()
vector_db.add_documents(all_text_chunks)

print(f"Vector database ready: {len(all_text_chunks)} chunks.")

# ============================================================
# PROMPTS
# ============================================================

SIMULATION_A_PROMPT = """
 You are a hospitality training assistant in Simulation A.

    Simulation A is focused on developing the student's judgment. Do not simply accept the student's reasoning.

    When the student gives a response:
    - Examine their reasoning.
    - Ask questions that make them think about their decision.
    - Point out important considerations they may have missed.
    - Challenge their assumptions when appropriate.
    - Provide constructive feedback.
    - If the student asks a question, answer it while helping them understand the reasoning behind the answer.

    Use the guest situation, previous conversation, and retrieved hospitality knowledge to guide the interaction.

    The goal is to help the student improve their decision-making, not simply give them the answer.

    Pay attention to noticeable spelling, grammar, and clarity issues in the student's response.
    If the student appears to have rushed their response or made a noticeable mistake, briefly encourage
    them to slow down and review their wording. Do not correct the mistake or provide the corrected version.
    Grammar and spelling feedback should support the hospitality training, not replace feedback about how well
    the student is addressing the guest's needs.

    When the student's response could better address the guest's needs, prioritize that feedback.
    If there is also a noticeable spelling or grammar issue, you may briefly mention both when appropriate.
    Do not force both types of feedback into every response.

    Make your feedback specific to the student's actual response and the guest's situation.
    Avoid repeatedly using the same wording or asking the same question when the student provides new input.
    Respond to what the student actually said and help them think through the next improvement.

    Keep each response concise, around 1-3 sentences.
    Do not explain every issue at once.
    Do not give the student the complete answer unless they have worked through the reasoning themselves.
    Encourage the student to try again when their response could be improved.
"""

GOOD_PROMPT = """
You are a hospitality training assistant in Simulation B.

    Your role is to help the student develop and improve a response to the guest.

    Start by providing a strong recommendation based on the guest's situation.

    After the student responds, treat their feedback as direction for improving the recommendation.
    Do not simply append their suggestion to the previous response. Reconsider and rewrite the
    recommendation so that the student's latest feedback meaningfully changes the response.

    When the student suggests a new action, concern, tone, or approach, incorporate it naturally
    into the recommendation and adjust the overall response when appropriate.
    For example, if the student asks you to offer water, do not simply add the words "offer water"
    to the previous recommendation. Rework the response so the additional gesture fits naturally
    into how the guest is being handled.

    If the student asks for a different tone, such as more caring, empathetic, personal, confident,
    or passionate, change the wording and approach of the recommendation to reflect that tone.
    Do not simply add a few emotional words to the existing response.

    Each new recommendation should feel like a genuine revision of the previous one, not the same
    response with one or two words added.

    Pay attention to the student's latest input and allow it to meaningfully influence the response.
    Relevant ideas from earlier in the conversation can be retained when appropriate, but do not
    preserve the previous wording or structure if the student's feedback calls for a different approach.

    Avoid repetitive wording, sentence structures, and recommendations.
    The response can be slightly longer when necessary to naturally reflect the student's feedback,
    but remain concise and appropriate for a hospitality training exercise.

    Do not argue with the student or evaluate whether their judgment is correct.
    The goal is to collaborate with the student and allow them to shape and improve the recommendation.

    Use the CATCH framework and retrieved hospitality knowledge when relevant.
    Do not invent hotel-specific details.

    Return only the updated recommendation.

"""

WEAK_PROMPT = """
You are a hospitality training assistant in Simulation B.
    Based on the guest's situation, the provided scenario, and the retrieved context,
    give the student one weak but plausible recommendation for how they should respond
    to or handle the guest.

    Focus primarily on the guest's immediate request and do not fully address underlying
    needs, concerns, or circumstances that may be present in the scenario. 

    Apply the CATCH framework only loosely and do not make a strong effort to incorporate 
    all relevant principles.

    After the student responds, treat their feedback as direction for revising the recommendation.
    Do not simply append their suggestion to the previous recommendation. Rewrite the recommendation
    so that the student's latest feedback meaningfully changes how the response is handled.

    If the student suggests a new action, concern, or tone, incorporate it naturally into a revised
    recommendation. The new response should reflect the student's input throughout the response
    rather than adding only a few words to the previous recommendation.

    If the student asks for more care, empathy, passion, personalization, or a different approach,
    change the wording and overall approach accordingly.

    The recommendation should remain weaker than an ideal hospitality response, but it should still
    respond meaningfully to the student's feedback. Do not make the recommendation weak by simply
    ignoring everything the student says.

    Avoid repeating the same wording and sentence structure across revisions.
    The response may be slightly longer when needed to naturally reflect the student's feedback.

    Do not argue with the student or evaluate whether their judgment is correct.

    Use general hospitality reasoning when appropriate, but do not invent hotel-specific policies,
    services, prices, timeframes, or other factual information that is not provided in the context.

    Return ONLY the updated practical recommendation.

"""

# ============================================================
# REQUEST MODEL
# ============================================================

class RecommendationRequest(BaseModel):
    guest_complaint: str = Field(min_length=1)
    simulation_type: str = "B"
    recommendation_type: str = "good"
    student_input: str | None = None
    student_question: str | None = None
    conversation_id: str | None = None
    new_conversation: bool = False


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================

def generate_recommendation(
    guest_complaint,
    simulation_type="B",
    student_input=None,
    student_question=None,
    recommendation_type="good",
    conversation_history=None,
):
    conversation_history = conversation_history or []

    retrieved_chunks = vector_db.search(
        guest_complaint,
        top_k=3
    )

    context_str = "\n\n---\n\n".join(retrieved_chunks)

    if simulation_type == "A":
        system_prompt = SIMULATION_A_PROMPT
    elif recommendation_type.lower() == "weak":
        system_prompt = WEAK_PROMPT
    else:
        system_prompt = GOOD_PROMPT

    system_prompt += (
        "\n\nRETRIEVED HOSPITALITY KNOWLEDGE:\n"
        + (context_str or "No relevant passages were retrieved.")
    )

    # The guest feedback is included in the current user message.
    # Only genuine follow-up requests should receive conversation history.
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for message in conversation_history:
        role = message.get("role")

        if role in ("user", "assistant"):
            messages.append({
                "role": role,
                "content": message.get("content", "")
            })

    user_parts = [
        "Guest feedback and scenario:",
        guest_complaint,
    ]

    if student_input and student_input.strip():
        user_parts.extend([
            "\nStudent response or requested revision:",
            student_input.strip(),
        ])

    if student_question and student_question.strip():
        user_parts.extend([
            "\nStudent's question:",
            student_question.strip(),
        ])

    messages.append({
        "role": "user",
        "content": "\n".join(user_parts),
    })

    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )

    answer = completion.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("The AI returned an empty recommendation.")

    return answer.strip()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "CATCH AI Recommendation API",
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        simulation_type = request.simulation_type.upper()

        if simulation_type not in ("A", "B"):
            raise HTTPException(
                status_code=400,
                detail="simulation_type must be A or B."
            )

        conversation_id = request.conversation_id

        # The frontend creates a fresh ID when starting a new session.
        # Always clear that ID when new_conversation is true.
        if request.new_conversation or not conversation_id:
            conversation_id = str(uuid.uuid4())

        if request.new_conversation:
            clear_conversation(conversation_id)

        # Unknown IDs can occur after a backend restart. They start empty.
        history = get_conversation(conversation_id)

        # The initial recommendation must not use old conversation history.
        # Follow-up requests use history only when this is NOT a new session.
        history_for_ai = [] if request.new_conversation else history

        recommendation = generate_recommendation(
            guest_complaint=request.guest_complaint,
            simulation_type=simulation_type,
            student_input=request.student_input,
            student_question=request.student_question,
            recommendation_type=request.recommendation_type,
            conversation_history=history_for_ai,
        )

        # Store a user turn and the assistant response for future follow-ups.
        # Save the guest feedback once at the beginning of the conversation.
        if request.new_conversation or not history:
            add_message(
                conversation_id,
                "user",
                "Guest feedback and scenario:\n"
                + request.guest_complaint
            )

        current_student_message = []

        if request.student_input and request.student_input.strip():
            current_student_message.append(
                request.student_input.strip()
            )

        if request.student_question and request.student_question.strip():
            current_student_message.append(
                request.student_question.strip()
            )

        if current_student_message:
            add_message(
                conversation_id,
                "user",
                "\n".join(current_student_message)
            )

        add_message(
            conversation_id,
            "assistant",
            recommendation
        )

        return {
            "recommendation": recommendation,
            "conversation_id": conversation_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Recommendation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a recommendation. Check the API logs."
        )
