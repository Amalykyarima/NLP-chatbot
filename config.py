# Configuration settings for our chatbot

MODEL_CONFIG = {
    # How similar the question should be to trigger a response
    "similarity_threshold": 0.6,
    "max_response_length": 500,
    "default_responses": [
        "I'm not sure how to respond to that.",
        "Could you please rephrase your question?",
        "I'm still learning. Can you ask something else?",
        "That's interesting! Tell me more."
    ]
}

CHAT_CONFIG = {
    "max_history": 10,
    "welcome_message": "Hello! I'm your NLP chatbot. How can I help you today?",
    "unknown_response": "I don't understand. Could you try asking differently?"
}
