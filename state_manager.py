
# Dictionary to keep track of user states
user_states = {}

def get_user_state(user_id):
    """Retrieve the current state of a user."""
    return user_states.get(user_id, {"intent": None, "context": None})

def set_user_state(user_id, intent, context):
    """Set the intent and context for a user."""
    user_states[user_id] = {"intent": intent, "context": context}

def clear_user_state(user_id):
    """Clear a user's state to reset the conversation."""
    if user_id in user_states:
        del user_states[user_id]
