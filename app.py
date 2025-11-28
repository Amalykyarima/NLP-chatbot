import streamlit as st
from chatbot import SimpleChatbot
from config import CHAT_CONFIG
import time

# Page configuration
st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize chatbot - this caches the bot so it doesn't reload every time


@st.cache_resource
def load_chatbot():
    return SimpleChatbot()


# Load our chatbot
chatbot = load_chatbot()

# App title and description
st.title("🤖 NLP Chatbot")
st.markdown("""
Welcome to your first NLP chatbot! This chatbot uses:
- **TF-IDF** for text processing
- **Cosine similarity** to understand your questions
- **Streamlit** for the beautiful interface
""")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": CHAT_CONFIG["welcome_message"]}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Add a small delay to make it feel more natural
            time.sleep(0.5)
            response = chatbot.get_response(prompt)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About This Chatbot")
    st.markdown("""
    **This chatbot understands:**
    - 👋 Greetings (hello, hi, hey)
    - 👋 Farewells (bye, goodbye)
    - ❓ Questions about its capabilities
    - 🌤️ Weather-related questions
    - 🙏 Thank you messages
    
    **How it works:**
    1. You type a message
    2. The bot cleans and processes your text
    3. It finds the most similar known pattern
    4. Returns an appropriate response
    """)

    # Conversation statistics
    st.subheader("📊 Conversation Stats")
    st.write(f"Total messages: {len(st.session_state.messages)}")
    st.write(f"Chatbot knowledge: {len(chatbot.qa_pairs)} categories")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": CHAT_CONFIG["welcome_message"]}
        ]
        st.rerun()

    st.markdown("---")
    st.markmary("**Tip:** Try asking 'What can you do?' or 'Hello!'")
