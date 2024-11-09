# dialogue_state_machine.py

class DialogueStateMachine:
    def __init__(self):
        # Dictionary to track conversation state per user
        self.user_states = {}

    def set_user_state(self, user_id, state):
        self.user_states[user_id] = state

    def get_user_state(self, user_id):
        return self.user_states.get(user_id, "default")

    def clear_user_state(self, user_id):
        if user_id in self.user_states:
            del self.user_states[user_id]
