# ============================================================
# IMPORTS
# ============================================================
# These are the libraries the API needs to run.
#
# FastAPI -> creates the API and the /recommend endpoint
# Pydantic -> defines what data the API expects from the backend
# fitz -> opens and reads the PDF files
# numpy -> does the vector/similarity calculations
# SentenceTransformer -> turns text into embeddings
# Groq -> lets us send the retrieved information + scenario
#         to the Qwen model
# os -> lets us get the Groq API key from Render's environment variables

import os
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq


# ============================================================
# CREATE THE API
# ============================================================
# This creates our FastAPI application.
#
# Once this is deployed to Render, this app will be running
# online and backend will be able to send requests to it.

app = FastAPI(title="CATCH AI Recommendation API")


# ============================================================
# GROQ SETUP
# ============================================================
# The Groq API key should NOT be written directly in this file.
#
# Locally, we can set it as an environment variable.
# On Render, we will add GROQ_API_KEY as a secret/environment variable.
#
# This keeps the API key private instead of putting it in GitHub.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# EMBEDDING MODEL
# ============================================================
# This is the same embedding model used in the notebook.
#
# It converts the PDF chunks and the guest scenario into
# numerical vectors.
#
# We need these vectors so the system can find the pieces of
# the CATCH/scenario documents that are most relevant to
# the guest situation.



# ============================================================
# PDF EXTRACTION + CHUNKING
# ============================================================
# This is the PDF processing function from the notebook.
#
# It does two things:
#
# 1. Extracts text from the PDFs.
# 2. Breaks that text into smaller chunks for the RAG system.
#
# If a PDF page is scanned/image-based and normal text extraction
# doesn't find enough text, it uses OCR instead.
#
# This means we don't have to manually turn every PDF page
# into screenshots.

def extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=100):

    print(f"Opening {pdf_path}...")

    doc = fitz.open(pdf_path)

    full_text = ""

    # Go through every page in the PDF
    for page_num, page in enumerate(doc):

        # First try to get normal selectable text
        text = page.get_text("text").strip()

        # If there is almost no text, assume the page is scanned
        # and use OCR to read the image instead.
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

        # Add this page's text to the complete document text
        full_text += text + "\n"

    doc.close()

    # --------------------------------------------------------
    # BREAK THE PDF TEXT INTO CHUNKS
    # --------------------------------------------------------
    # The entire PDF shouldn't be sent to the model every time.
    #
    # Instead, we split it into smaller pieces.
    # Later, the vector database will find the most relevant
    # chunks for the specific guest scenario.

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


# ============================================================
# SIMPLE VECTOR DATABASE
# ============================================================
# This is the same basic vector database from the notebook.
#
# Its job is to store:
#
# - the text chunks from our PDFs
# - the numerical embeddings for those chunks
#
# When a guest scenario comes in, search() finds the chunks
# that are most similar/relevant to that scenario.
vectorizer = TfidfVectorizer()
class SimpleVectorDB:

    def __init__(self):

        self.chunks = []
        self.embeddings = []

    # Add all of our PDF chunks to the database
    def add_documents(self, text_chunks, embedding_model):

        self.chunks.extend(text_chunks)

        # Convert every chunk into an embedding/vector
        vectors = vectorizer.transform(text_chunks).toarray()
       

        if len(self.embeddings) == 0:

            self.embeddings = vectors

        else:

            self.embeddings = np.vstack(
                (self.embeddings, vectors)
            )

    # Find the chunks that are most relevant to a query
    def search(
        self,
        query,
        embedding_model,
        top_k=3
    ):

        # Convert the guest scenario into an embedding
        query_vector = vectorizer.transform([query]).toarray()[0]

        # Compare the guest scenario to every PDF chunk
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

        # Calculate cosine similarity
        similarities = (
            dot_products /
            (norm_db * norm_query)
        )

        # Get the three most relevant chunks
        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        return [
            self.chunks[index]
            for index in top_indices
        ]


# ============================================================
# BUILD THE KNOWLEDGE BASE
# ============================================================
# This happens when the API starts.
#
# We process both PDFs:
#
# - CATCH framework
# - scenario information
#
# Then we combine all of their chunks into the same
# vector database.
#
# This is what allows the AI to use BOTH sources when
# creating a recommendation.

vector_db = SimpleVectorDB()

pdf_files = [
    "catch_framwork.pdf",
    "scenarios.pdf"
]

all_text_chunks = []

for pdf_file in pdf_files:

    # Make sure the PDF was actually included in the
    # project before trying to process it.
    if not os.path.exists(pdf_file):

        raise RuntimeError(
            f"Required PDF not found: {pdf_file}"
        )

    chunks = extract_and_chunk_pdf(pdf_file)

    # Add this PDF's chunks to the combined list
    all_text_chunks.extend(chunks)


print(
    f"Total chunks from all PDFs: "
    f"{len(all_text_chunks)}"
)

vectorizer.fit(all_text_chunks)
# Turn all chunks into embeddings and store them
vector_db.add_documents(all_text_chunks, vectorizer)

print("Vector database is ready.")


# ============================================================
# GOOD RECOMMENDATION PROMPT
# ============================================================
# This is the prompt for the "good" AI behavior.
#
# The model should:
# - understand the scenario
# - identify the immediate request
# - notice underlying needs/concerns
# - use the CATCH framework
# - give ONE practical recommendation
#
# We don't want the model explaining its reasoning to the student.

GOOD_PROMPT = (
    "You are a hospitality training assistant. "
    "Based on the guest's situation, the provided scenario, and the retrieved context, "
    "determine what the situation requires and give the student one clear, practical recommendation "
    "for how they should respond to or handle the guest. "
    "Use the scenario to identify the guest's immediate request as well as any underlying needs, "
    "concerns, or circumstances that should affect how the student responds. "
    "Apply the CATCH framework: Care, Adaptability, Think, "
    "Create Exceptional Experiences, and Human Connection. "
    "Use the provided context and scenario details to guide the recommendation. "
    "Use general hospitality reasoning when appropriate, but do not invent hotel-specific policies, "
    "services, prices, timeframes, or other factual information that is not provided in the context. "
    "Do not provide analysis, reasoning, explanations, or a breakdown of the CATCH framework. "
    "Return ONLY the practical recommendation that the student should follow. "
    "Keep the recommendation concise, specific, professional, and actionable."
)


# ============================================================
# WEAK RECOMMENDATION PROMPT
# ============================================================
# This is the second possible AI behavior.
#
# The recommendation should NOT be obviously ridiculous.
# It should be plausible, but weaker than the good recommendation.
#
# For example, it might address the guest's immediate request
# but fail to recognize the underlying concern.

WEAK_PROMPT = (
    "You are a hospitality training assistant. "
    "Based on the guest's situation, the provided scenario, and the retrieved context, "
    "give the student one weak but plausible recommendation for how they should respond "
    "to or handle the guest. "
    "Focus primarily on the guest's immediate request and do not fully address underlying "
    "needs, concerns, or circumstances that may be present in the scenario. "
    "Apply the CATCH framework only loosely and do not make a strong effort to incorporate "
    "all relevant principles. "
    "The recommendation may be generic, reactive, or minimally helpful, while still being "
    "reasonable enough that a student could plausibly give this response in a hospitality setting. "
    "Use general hospitality reasoning when appropriate, but do not invent hotel-specific policies, "
    "services, prices, timeframes, or other factual information that is not provided in the context. "
    "Do not provide analysis, reasoning, explanations, or a breakdown of the CATCH framework. "
    "Return ONLY the practical recommendation that the student should follow. "
    "Keep the recommendation concise and professional."
)


# ============================================================
# DEFINE WHAT THE API RECEIVES
# ============================================================
# This tells FastAPI what information backend can send.
#
# scenario -> the guest situation
#
# student_response -> optional. This allows the API to also
#                     receive what the student said/did.
#
# recommendation_type -> tells our API whether we want the
#                        GOOD or WEAK recommendation.
#
#  randomly choose "good" or "weak"
# instead of the frontend deciding.

class RecommendationRequest(BaseModel):

    scenario: str

    student_response: str = ""

    recommendation_type: str = "good"


# ============================================================
# GENERATE THE RECOMMENDATION
# ============================================================
# This is the main AI pipeline.
#
# The flow is:
#
# Guest scenario
#      ↓
# Vector search
#      ↓
# Relevant CATCH/scenario chunks
#      ↓
# Good or weak system prompt
#      ↓
# Qwen through Groq
#      ↓
# Recommendation
#
# This function connects the RAG part of the project to the
# actual AI model.

def generate_recommendation(
    scenario,
    student_response="",
    recommendation_type="good"
):

    # --------------------------------------------------------
    # STEP 1: FIND RELEVANT INFORMATION
    # --------------------------------------------------------
    # Search the PDFs for the three chunks most relevant
    # to this guest scenario.

    retrieved_chunks = vector_db.search(
    scenario,
    vectorizer,
    top_k=3
    )

    # Combine those chunks into one context string
    context_str = "\n\n---\n\n".join(
        retrieved_chunks
    )


    # --------------------------------------------------------
    # STEP 2: CHOOSE GOOD OR WEAK PROMPT
    # --------------------------------------------------------
    # The backend tells us which type of recommendation
    # we want.

    if recommendation_type == "weak":

        system_prompt = WEAK_PROMPT

    else:

        system_prompt = GOOD_PROMPT


    # Add the RAG information to the selected prompt
    system_prompt += (
        f"\n\nRETRIEVED CONTEXT:\n"
        f"{context_str}"
    )


    # --------------------------------------------------------
    # STEP 3: CREATE THE USER MESSAGE
    # --------------------------------------------------------
    # This is the actual guest situation being given to Qwen.

    user_message = (
        f"Guest scenario:\n{scenario}\n"
    )

    # Include the student's response if one was provided
    if student_response:

        user_message += (
            f"\nStudent response:\n"
            f"{student_response}\n"
        )


    # --------------------------------------------------------
    # STEP 4: CALL GROQ / QWEN
    # --------------------------------------------------------
    # Groq sends our prompt to the Qwen model.
    #
    # temperature=0.0 makes the output more consistent.
    # max_tokens=800 limits how long the response can be.

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


    # Return only the actual recommendation text
    return completion.choices[0].message.content


# ============================================================
# TEST / HOME ROUTE
# ============================================================
# This isn't the AI endpoint.
#
# It's just a simple way to check whether the API is online.
#
# If you visit the API's main URL and see this response,
# you know the server is running.

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "CATCH AI Recommendation API"
    }


# ============================================================
# RECOMMENDATION API ENDPOINT
# ============================================================
# THIS is the endpoint backend will call.
#
# The backend sends a POST request to:
#
# /recommend
#
# with the scenario, student response, and recommendation type.
#
# The endpoint then runs our RAG + AI pipeline and sends the
# recommendation back as JSON.

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    try:

        recommendation = generate_recommendation(

            scenario=request.scenario,

            student_response=request.student_response,

            recommendation_type=request.recommendation_type
        )

        # Send the AI result back to backend
        return {
            "recommendation": recommendation
        }

    except Exception as e:

        # If something goes wrong, return an API error
        # instead of crashing silently.
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )