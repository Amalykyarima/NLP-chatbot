import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections
import logging
from datetime import datetime
import sys
import os

from nlp_processor import NLPProcessor

# Configure page
st.set_page_config(
    page_title="MSc Program FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Minimum cosine-similarity score for a TF-IDF match to be considered confident
# enough to answer with. Below this, we fall back to the small-talk chatbot.
SIMILARITY_THRESHOLD = 0.3

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


# -----------------------------------------------------------------------
# FAQ knowledge base
#
# Each entry has a canonical question plus a list of alternate phrasings
# that should all resolve to the same answer. The TF-IDF/cosine-similarity
# matcher compares the user's message against every phrasing and returns
# the answer for whichever phrasing scores highest, so long as it clears
# SIMILARITY_THRESHOLD.
# -----------------------------------------------------------------------
FAQ_DATABASE = [
    {
        "questions": [
            "What courses are offered in the MSc program?",
            "What courses are offered?",
            "courses offered",
            "what courses",
            "list courses",
            "course information",
        ],
        "answer": (
            "📚 COURSE INFORMATION:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "The MSc program offers the following courses:\n"
            "• Advanced Algorithms\n"
            "• Machine Learning\n"
            "• Cybersecurity\n"
            "• Data Science\n"
            "• Artificial Intelligence\n"
            "• Natural Language Processing\n"
            "• Computer Vision\n"
            "• Big Data Analytics\n\n"
            "Each course is worth 15 credits and runs for one semester.\n"
            "For detailed course descriptions, visit: www.university.edu/courses"
        ),
    },
    {
        "questions": [
            "How can I enroll in a course?",
            "How can I enroll?",
            "How do I enroll",
            "enroll in course",
            "register for course",
            "enrollment procedures",
            "enrolment procedures",
        ],
        "answer": (
            "📝 ENROLLMENT PROCEDURES:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Step 1: Visit the enrollment portal at www.university.edu/enrollment\n"
            "Step 2: Log in with your student ID and password\n"
            "Step 3: Browse available courses for the semester\n"
            "Step 4: Select courses (maximum 4 per semester)\n"
            "Step 5: Confirm your selections\n"
            "Step 6: Pay tuition fees (deadline: August 25th)\n\n"
            "⚠️ Note: You must complete enrollment before August 25th.\n"
            "💡 Tip: Speak with your academic advisor before enrolling!"
        ),
    },
    {
        "questions": [
            "When does the semester start?",
            "semester start date",
            "start date",
            "important dates",
            "registration deadline",
            "What is the deadline for course registration?",
        ],
        "answer": (
            "📅 IMPORTANT DATES:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 Course Registration Opens: August 1st, 2025\n"
            "📌 Course Registration Deadline: August 25th, 2025\n"
            "📌 Tuition Payment Deadline: August 25th, 2025\n"
            "📌 Semester Start Date: September 1st, 2025\n"
            "📌 Add/Drop Period: September 1st - September 15th\n"
            "📌 Mid-term Exams: October 20th - October 27th\n"
            "📌 Final Exams: December 10th - December 20th\n"
            "📌 Semester End Date: December 20th, 2025\n\n"
            "💡 Mark your calendar! Late registration incurs a £50 fee."
        ),
    },
    {
        "questions": [
            "Who should I contact for academic advising?",
            "academic advisor contact",
            "advising contact",
            "contact information",
            "who to contact",
        ],
        "answer": (
            "📞 CONTACT INFORMATION:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "👤 Academic Advisor:\n"
            "   Dr. Jane Smith\n"
            "   📧 jane.smith@example.com\n"
            "   ☎️ +44 123 456 7890\n"
            "   🏢 Office: Building A, Room 205\n"
            "   🕐 Office Hours: Mon-Fri, 10:00 AM - 4:00 PM\n\n"
            "🏫 Program Coordinator:\n"
            "   Prof. John Doe\n"
            "   📧 john.doe@example.com\n"
            "   ☎️ +44 123 456 7891\n\n"
            "📚 Student Services:\n"
            "   📧 studentservices@example.com\n"
            "   ☎️ +44 123 456 7892\n\n"
            "💰 Finance Office:\n"
            "   📧 finance@example.com\n"
            "   ☎️ +44 123 456 7893"
        ),
    },
]


# Small-talk / control patterns handled by NLTK's rule-based Chat utility.
# These are intentionally kept separate from the FAQ knowledge base above:
# greetings, thanks, help and exit don't need semantic matching, and routing
# them through the same TF-IDF space would just dilute the FAQ vectors.
SMALL_TALK_PAIRS = [
    (r'Hi|Hello|Hey|Greetings|Good morning|Good afternoon|Good evening|how are you|how are you doing',
     ['Hello! Welcome to the MSc Program FAQ Bot. How can I help you today?',
      'Hi there! I\'m here to answer your questions about the MSc program.',
      'Hey! Ask me anything about courses, enrollment, or program details.']),

    (r'Help|help|what can you do|menu|options|whatsup',
     ['You can ask me about:\n\n'
      '📚 1. Course information - Type: "course information"\n'
      '📝 2. Enrollment procedures - Type: "enrollment procedures"\n'
      '📅 3. Important dates - Type: "important dates"\n'
      '📞 4. Contact information - Type: "contact information"\n\n'
      'Or ask specific questions like:\n'
      '• "What courses are offered?"\n'
      '• "How can I enroll?"\n'
      '• "When does the semester start?"\n'
      '• "Who should I contact for advising?"']),

    (r'thank you|thanks|thank|appreciate',
     ['You\'re welcome! Happy to help!',
      'Glad I could assist you!',
      'Anytime! Feel free to ask more questions.']),

    (r'exit|quit|bye|goodbye|see you',
     ['Goodbye! Have a great day! Feel free to come back anytime.',
      'Thanks for chatting! Good luck with your studies!',
      'See you later! Don\'t hesitate to return if you have more questions.']),
]


@st.cache_resource
def setup_small_talk_chat():
    """Rule-based fallback chat for greetings/help/thanks/exit."""
    return Chat(SMALL_TALK_PAIRS, reflections)


@st.cache_resource
def setup_nlp_processor():
    """
    Build and fit the TF-IDF NLPProcessor against every phrasing in the FAQ
    database, so user messages can be semantically matched to the closest
    known question rather than requiring an exact regex hit.
    """
    processor = NLPProcessor()
    all_phrasings = [q for entry in FAQ_DATABASE for q in entry["questions"]]
    processor.fit_vectorizer(all_phrasings)
    return processor


def get_faq_answer(user_message, processor):
    """
    Compare the user's message against every known FAQ phrasing using
    TF-IDF cosine similarity, and return the answer for the best match
    if it clears SIMILARITY_THRESHOLD. Returns (answer, score) so callers
    can log/inspect the confidence if needed; answer is None on no match.
    """
    best_score = 0.0
    best_answer = None

    for entry in FAQ_DATABASE:
        for phrasing in entry["questions"]:
            score = processor.get_similarity(user_message, phrasing)
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_answer, best_score
    return None, best_score


def get_response(user_message, processor, small_talk_chat):
    """
    Main response router:
    1. Try semantic FAQ matching via TF-IDF/cosine similarity.
    2. Fall back to rule-based small talk (greetings, thanks, help, exit).
    3. Fall back to a generic "I don't understand" message.
    """
    faq_answer, score = get_faq_answer(user_message, processor)
    if faq_answer:
        return faq_answer

    small_talk_response = small_talk_chat.respond(user_message)
    if small_talk_response and small_talk_response != "None":
        return small_talk_response

    return (
        "I'm sorry, I don't understand that question. Type 'Help' to see what I can answer, or try one of these:\n"
        "• 'course information'\n"
        "�� 'enrollment procedures'\n"
        "• 'important dates'\n"
        "• 'contact information'"
    )


def main():
    # Download NLTK data
    download_nltk_data()

    # Initialize the TF-IDF FAQ matcher and the small-talk fallback chatbot
    processor = setup_nlp_processor()
    small_talk_chat = setup_small_talk_chat()

    # App title and description
    st.title("🎓 MSc Program FAQ Chatbot")
    st.markdown("""
    Welcome to your MSc Program assistant! I can help you with:
    - 📚 Course information
    - 📝 Enrollment procedures
    - 📅 Important dates
    - 📞 Contact information
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "👋 Hello! Welcome to the MSc Program FAQ Bot. How can I help you today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about courses, enrollment, dates, or contacts..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response via TF-IDF FAQ matching, falling back to small talk
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt, processor, small_talk_chat)
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

    # Sidebar with additional information
    with st.sidebar:
        st.header("ℹ️ About This Chatbot")
        st.markdown("""
        **This chatbot can help you with:**

        **📚 Course Information**
        - Available courses
        - Course descriptions
        - Credit information

        **📝 Enrollment**
        - Registration steps
        - Enrollment deadlines
        - Payment information

        **📅 Important Dates**
        - Semester start/end dates
        - Registration deadlines
        - Exam schedules

        **📞 Contacts**
        - Academic advisors
        - Program coordinators
        - Student services
        """)

        # Conversation statistics
        st.subheader("📊 Conversation Stats")
        st.write(f"Total messages: {len(st.session_state.messages)}")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "👋 Hello! Welcome to the MSc Program FAQ Bot. How can I help you today?"}
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("**💡 Tip:** Try typing 'Help' to see all available topics!")


if __name__ == "__main__":
    main()
