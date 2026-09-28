# CONVERSATION MEMORY

# This file is responsible only for conversation storage.
#
# It does NOT handle:
# - FastAPI
# - Groq
# - Qwen
# - PDFs
# - RAG
# - AI prompts
#
# It simply creates conversation IDs and stores messages
# associated with those conversations.

import uuid


# Stores all active conversations in memory.
#
# Example:
#
# {
#     "conversation-id-123": [
#         {"role": "guest", "content": "..."},
#         {"role": "student", "content": "..."},
#         {"role": "assistant", "content": "..."}
#     ]
# }

conversations = {}


# CREATE A NEW CONVERSATION

def create_conversation():
    """
    Creates a new conversation and returns its ID.
    """

    conversation_id = str(uuid.uuid4())

    conversations[conversation_id] = []

    return conversation_id


# GET CONVERSATION

def get_conversation(conversation_id):
    """
    Returns the messages belonging to a conversation.
    """

    return conversations.get(
        conversation_id,
        []
    )


# ADD MESSAGE

def add_message(conversation_id, role, content):
    """
    Adds a message to an existing conversation.

    If the conversation does not exist yet,
    it creates it automatically.
    """

    if conversation_id not in conversations:
        conversations[conversation_id] = []

    conversations[conversation_id].append({
        "role": role,
        "content": content
    })