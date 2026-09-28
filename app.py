# IMPORTS

# These are the libraries the API needs to run.
#
# FastAPI -> creates the API and the /recommend endpoint
# Pydantic -> defines what data the API expects
# fitz -> opens and reads the PDF files
# numpy -> does the vector/similarity calculations
# TfidfVectorizer -> turns text into numerical TF-IDF vectors
# Groq -> sends the retrieved information + scenario to Qwen
# os -> gets the Groq API key from environment variables
# conversation -> manages conversation memory

import os
import fitz
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI, HTTPException, Request, Response

from pydantic import BaseModel

from groq import Groq

from conversation import (
    create_conversation,
    get_conversation,
    add_message
)


# CREATE THE API

app = FastAPI(
    title="CATCH AI Recommendation API"
)


# GROQ SETUP

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# PDF EXTRACTION + CHUNKING

def extract_and_chunk_pdf(
    pdf_path,
    chunk_size=500,
    overlap=100
):

    print(f"Opening {pdf_path}...")

    doc = fitz.open(pdf_path)

    full_text = ""

    # Go through every page in the PDF
    for page_num, page in enumerate(doc):

        # Try normal selectable text first
        text = page.get_text("text").strip()

        # If there is almost no text, try OCR
        if len(text) < 50:

            print(
                f" -> Page {page_num + 1} appears to be an image. "
                "Running OCR..."
            )

            try:

                ocr_textpage = page.get_textpage_ocr(
                    flags=3,
                    language="eng"
                )

                text = page.get_text(
                    "text",
                    textpage=ocr_textpage
                )

            except Exception as e:

                print(
                    f"OCR failed on page {page_num + 1}: {e}"
                )

                text = ""

        else:

            print(
                f" -> Page {page_num + 1} extracted "
                "via standard digital text."
            )

        full_text += text + "\n"

    doc.close()

    # BREAK THE PDF TEXT INTO CHUNKS

    words = full_text.split()

    chunks = []

    word_chunk_size = chunk_size // 5
    word_overlap = overlap // 5

    step = word_chunk_size - word_overlap

    for i in range(0, len(words), step):

        chunk_words = words[
            i:i + word_chunk_size
        ]

        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 10:
            chunks.append(chunk_text)

    print(
        f"Created {len(chunks)} chunks from {pdf_path}"
    )

    return chunks


# SIMPLE VECTOR DATABASE

vectorizer = TfidfVectorizer()


class SimpleVectorDB:

    def __init__(self):

        self.chunks = []
        self.embeddings = []

    # Add PDF chunks to the database
    def add_documents(self, text_chunks):

        self.chunks.extend(text_chunks)

        vectors = vectorizer.transform(
            text_chunks
        ).toarray()

        if len(self.embeddings) == 0:

            self.embeddings = vectors

        else:

            self.embeddings = np.vstack(
                (
                    self.embeddings,
                    vectors
                )
            )

    # Find the most relevant chunks
    def search(
        self,
        query,
        top_k=3
    ):

        query_vector = vectorizer.transform(
            [query]
        ).toarray()[0]

        dot_products = np.dot(
            self.embeddings,
            query_vector
        )

        norm_db = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        norm_query = np.linalg.norm(
            query_vector
        )

        similarities = (
            dot_products /
            (norm_db * norm_query)
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        return [
            self.chunks[index]
            for index in top_indices
        ]


# BUILD THE KNOWLEDGE BASE

vector_db = SimpleVectorDB()

pdf_files = [
    "catch_framwork.pdf",
    "scenarios.pdf"
]

all_text_chunks = []

for pdf_file in pdf_files:

    if not os.path.exists(pdf_file):

        raise RuntimeError(
            f"Required PDF not found: {pdf_file}"
        )

    chunks = extract_and_chunk_pdf(
        pdf_file
    )

    all_text_chunks.extend(chunks)


print(
    f"Total chunks from all PDFs: "
    f"{len(all_text_chunks)}"
)

vectorizer.fit(
    all_text_chunks
)

vector_db.add_documents(
    all_text_chunks
)

print(
    "Vector database is ready."
)


# SIMULATION A PROMPT

SIMULATION_A_PROMPT = (
     """
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
"""
)



# GOOD RECOMMENDATION PROMPT

GOOD_PROMPT = (
     """
    You are a hospitality training assistant in Simulation B.

    Your role is to help the student develop their own response to the guest.

    Start by providing a good recommendation based on the guest's situation.

    After the student responds, continuously adjust the recommendation based on what the student says. If the student rejects the previous recommendation, do not defend or repeat it. Adapt to their reasoning and create a new recommendation that reflects their input, even if the previous recommendation was already a good response.

    Continue adjusting the recommendation each time the student provides new suggestions, observations, or changes in direction. Do not stop adapting just because the previous recommendation was reasonable or effective.

    The student's latest input should influence the next recommendation, while relevant ideas from earlier in the conversation can be retained when appropriate.

    Do not argue with the student or evaluate whether their judgment is correct. The goal is to collaborate with the student and help turn their ideas into a practical guest response.

    Use the CATCH framework and retrieved hospitality knowledge when relevant.
    Do not invent hotel-specific details.

    Return only the updated recommendation.
    Keep it concise, around 1-2 sentences.
    """

    # "You are a hospitality training assistant. "
    # "Based on the guest's situation, the provided scenario, and the retrieved context, "
    # "give the student one clear, practical recommendation for how they should respond to or handle the guest. "
    # "The recommendation should address the guest's immediate request while naturally responding to any underlying need or concern revealed by the scenario. "
    # "Apply the CATCH framework: Care, Adaptability, Think, Create Exceptional Experiences, and Human Connection. "
    # "Do not explicitly describe, name, or explain the guest's underlying concern or the reasoning behind the recommendation. "
    # "Instead, express the recommendation naturally as something the student could actually say or do with the guest. "
    # "Do not use phrases such as 'acknowledge the underlying concern,' 'recognize the burden,' "
    # "'identify the guest's emotional need,' or similar instructional language. "
    # "Use general hospitality reasoning when appropriate, but do not invent hotel-specific policies, "
    # "services, prices, timeframes, or other factual information that is not provided in the context. "
    # "Return ONLY the practical recommendation. "
    # "Keep it concise, specific, natural, professional, and actionable, ideally one or two sentences."
)



# WEAK RECOMMENDATION PROMPT


WEAK_PROMPT = (
    "You are a hospitality training assistant in Simulation B. "
    "Based on the guest's situation, the provided scenario, and the retrieved context, "
    "give the student one weak but plausible recommendation for how they should respond "
    "to or handle the guest. "

    "Focus primarily on the guest's immediate request and do not fully address underlying "
    "needs, concerns, or circumstances that may be present in the scenario. "

    "Apply the CATCH framework only loosely and do not make a strong effort to incorporate "
    "all relevant principles. "

    "The recommendation may be generic, reactive, or minimally helpful, while still being "
    "reasonable enough that a student could plausibly give this response in a hospitality setting. "

    "After the student responds, continuously adjust the recommendation based on what the student says. "
    "If the student rejects the previous recommendation, do not defend or repeat it. "
    "Adapt to the student's input and create a new recommendation based on what they are asking for, "
    "even if the previous recommendation was already reasonable. "

    "Continue adjusting the recommendation each time the student provides new suggestions, "
    "observations, or changes in direction. The student's latest input should influence the next "
    "recommendation, while relevant ideas from earlier in the conversation can be retained when appropriate. "

    "Do not argue with the student or evaluate whether their judgment is correct. "

    "Use general hospitality reasoning when appropriate, but do not invent hotel-specific policies, "
    "services, prices, timeframes, or other factual information that is not provided in the context. "

    "Do not provide analysis, reasoning, explanations, or a breakdown of the CATCH framework. "

    "Return ONLY the updated practical recommendation that the student should follow. "
    "Keep the recommendation concise and professional."
)



# DEFINE WHAT THE API RECEIVES

#
# Notice that conversation_id is NOT here.
#
# The backend handles the conversation ID through a cookie.
#
# The frontend only needs to send the information it actually
# knows about:
#
# - guest_complaint
# - simulation_type
# - recommendation_type
# - student_input
# - student_question

class RecommendationRequest(BaseModel):

    guest_complaint: str
    simulation_type: str = "B"
    recommendation_type: str = "good"
    student_input: str | None = None
    student_question: str | None = None


# GENERATE THE RECOMMENDATION

def generate_recommendation(
    guest_complaint,
    simulation_type="B",
    student_input=None,
    student_question=None,
    recommendation_type="good",
    conversation_history=None
):

    if conversation_history is None:
        conversation_history = []


    # STEP 1: FIND RELEVANT INFORMATION

    retrieved_chunks = vector_db.search(
        guest_complaint,
        top_k=3
    )

    context_str = "\n\n---\n\n".join(
        retrieved_chunks
    )


    # STEP 2: FORMAT CONVERSATION HISTORY

    # This is what allows the AI to remember previous turns.

    history_str = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )


    # STEP 3: CHOOSE PROMPT

    if simulation_type == "A":

        system_prompt = SIMULATION_A_PROMPT

    elif recommendation_type == "weak":

        system_prompt = WEAK_PROMPT

    else:

        system_prompt = GOOD_PROMPT


    # Add RAG context
    system_prompt += (
        f"\n\nRETRIEVED CONTEXT:\n"
        f"{context_str}"
    )


    # STEP 4: CREATE USER MESSAGE

    if simulation_type == "A":

        user_message = (
            f"Guest interaction:\n"
            f"{guest_complaint}\n"
        )
    else:

        user_message = (
            f"Guest complaint:\n"
            f"{guest_complaint}\n"
        )

    # Add previous conversation
    if history_str:

        user_message += (
            f"\nPrevious conversation:\n"
            f"{history_str}\n"
        )


    # Add current student input
    if student_input:

        user_message += (
            f"\nStudent input:\n"
            f"{student_input}\n"
        )

    # Add current student question
    if student_question:

        user_message += (
            f"\nStudent question:\n"
            f"{student_question}\n"
        )


    # STEP 5: CALL GROQ / QWEN

    completion = groq_client.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        temperature=0.0,

        max_tokens=800
    )


    return completion.choices[0].message.content


# TEST / HOME ROUTE

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "CATCH AI Recommendation API"
    }


# RECOMMENDATION API ENDPOINT

#
# The frontend does NOT send a conversation ID.
#
# Instead:
#
# 1. The backend checks for a conversation cookie.
# 2. If there isn't one, the backend creates a conversation.
# 3. The backend gets the conversation history.
# 4. The AI uses that history.
# 5. The new messages are saved.
# 6. The backend sends the conversation cookie back.
#
# The browser can then automatically send that cookie on
# future requests.

@app.post("/recommend")
def recommend(
    request: RecommendationRequest,
    http_request: Request,
    response: Response
):

    try:

        # GET OR CREATE CONVERSATION

        conversation_id = (
            http_request.cookies.get(
                "conversation_id"
            )
        )

        if not conversation_id:

            conversation_id = (
                create_conversation()
            )

            # Tell the browser to remember the ID.
            #
            # The frontend does not need to create or manage it.

            response.set_cookie(
                key="conversation_id",
                value=conversation_id,
                httponly=True,
                secure=False,
                samesite="lax"
            )

        # GET EXISTING CONVERSATION

        history = get_conversation(
            conversation_id
        )

        # SAVE ORIGINAL GUEST COMPLAINT
        #
        # Only save it if this is the beginning of a
        # conversation.

        if not history:

            add_message(
                conversation_id,
                "guest",
                request.guest_complaint
            )

            history = get_conversation(
                conversation_id
            )


        # GENERATE AI RESPONSE

        recommendation = generate_recommendation(

            guest_complaint=request.guest_complaint,

            simulation_type=request.simulation_type,

            student_input=request.student_input,

            student_question=request.student_question,

            recommendation_type=request.recommendation_type,

            conversation_history=history
        )


        # SAVE STUDENT INPUT

        if request.student_input:

            add_message(
                conversation_id,
                "student",
                request.student_input
            )


        # SAVE STUDENT QUESTION

        if request.student_question:

            add_message(
                conversation_id,
                "student",
                request.student_question
            )


        # SAVE AI RESPONSE

        add_message(
            conversation_id,
            "assistant",
            recommendation
        )


        # RETURN RESPONSE

        return {
            "recommendation": recommendation
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )