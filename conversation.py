
# conversation.py
# Stores conversation history in server memory.
# A backend restart clears all conversations automatically.

import uuid
from threading import RLock

conversations = {}
_conversations_lock = RLock()


def create_conversation():
    """Create a new, empty conversation."""
    conversation_id = str(uuid.uuid4())

    with _conversations_lock:
        conversations[conversation_id] = []

    return conversation_id


def get_conversation(conversation_id):
    """Return a copy of the conversation history."""
    if not conversation_id:
        return []

    with _conversations_lock:
        return [
            message.copy()
            for message in conversations.get(conversation_id, [])
        ]


def add_message(conversation_id, role, content):
    """Add a message to an existing conversation."""
    if not conversation_id or not content:
        return

    with _conversations_lock:
        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append({
            "role": role,
            "content": content
        })


def clear_conversation(conversation_id):
    """Clear history for a conversation without reusing old messages."""
    if not conversation_id:
        return

    with _conversations_lock:
        conversations[conversation_id] = []


def delete_conversation(conversation_id):
    """Remove a conversation completely."""
    if not conversation_id:
        return

    with _conversations_lock:
        conversations.pop(conversation_id, None)
