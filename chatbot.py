import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections
import logging
from datetime import datetime
import sys
import os

# Configure page
st.set_page_config(
    page_title="MSc Program FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Download required NLTK data


@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Initialize the chatbot


@st.cache_resource
def setup_chatbot():
    # Define the pairs of patterns and responses (FAQ Database)
    pairs = [
        # Course Information
        (r'What courses are offered in the MSc program?|courses offered|what courses|courses|course',
         ['The MSc program offers courses in Advanced Algorithms, Machine Learning, Cybersecurity, Data Science, Artificial Intelligence, and more.']),

        (r'How can I enroll in a course?|enroll|enrollment|register for course',
         ['You can enroll in a course by visiting the course enrollment page on the university website at www.university.edu/enrollment']),

        # Contact Information
        (r'Who should I contact for academic advising?|academic advisor|advising contact',
         ['You can contact the academic advisor, Dr. Jane Smith, at jane.smith@example.com or call +44 123 456 7890.']),

        # Important Dates
        (r'When does the semester start?|semester start|start date',
         ['The semester starts on September 1st, 2025.']),

        (r'What is the deadline for course registration?|registration deadline|deadline',
         ['The deadline for course registration is August 25th, 2025.']),

        # Greetings
        (r'Hi|Hello|Hey|Greetings|how are you|how are you doing',
         ['Hello! Welcome to the MSc Program FAQ Bot. How can I help you today?',
          'Hi there! I\'m here to answer your questions about the MSc program.',
          'Hey! Ask me anything about courses, enrollment, or program details.']),

        # Help Command
        (r'Help|help|what can you do|what|whatsup',
         ['You can ask me about:\n1. Courses offered in the MSc program\n2. Course enrollment procedures\n3. Contact information for academic advising\n4. Semester start dates\n5. Course registration deadlines\n6. General MSc program information']),

        # Exit Commands
        (r'exit|quit|bye|goodbye',
         ['Goodbye! Have a great day! Feel free to come back anytime.',
          'Thanks for chatting! Good luck with your studies!',
          'See you later! Don\'t hesitate to return if you have more questions.']),

        # Catch-all for unknown questions
        (r'(.*)',
         ["I'm sorry, I don't understand that question. Type 'Help' to see what I can answer, or try rephrasing your question."]),

        # MENU OPTION RESPONSES - When user types exact menu items
        (r'course information|courses information',
         ['📚 COURSE INFORMATION:\n'
          '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
          'The MSc program offers the following courses:\n'
          '• Advanced Algorithms\n'
          '• Machine Learning\n'
          '• Cybersecurity\n'
          '• Data Science\n'
          '• Artificial Intelligence\n'
          '• Natural Language Processing\n'
          '• Computer Vision\n'
          '• Big Data Analytics\n\n'
          'Each course is worth 15 credits and runs for one semester.\n'
          'For detailed course descriptions, visit: www.university.edu/courses']),

        (r'Course enrollment procedures|enrolment procedures|enrollment|enrolment',
         ['📝 ENROLLMENT PROCEDURES:\n'
          '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
          'Step 1: Visit the enrollment portal at www.university.edu/enrollment\n'
          'Step 2: Log in with your student ID and password\n'
          'Step 3: Browse available courses for the semester\n'
          'Step 4: Select courses (maximum 4 per semester)\n'
          'Step 5: Confirm your selections\n'
          'Step 6: Pay tuition fees (deadline: August 25th)\n\n'
          '⚠️ Note: You must complete enrollment before August 25th.\n'
          '💡 Tip: Speak with your academic advisor before enrolling!']),

        (r'important dates|dates|registration',
         ['📅 IMPORTANT DATES:\n'
          '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
          '📌 Course Registration Opens: August 1st, 2025\n'
          '📌 Course Registration Deadline: August 25th, 2025\n'
          '📌 Tuition Payment Deadline: August 25th, 2025\n'
          '📌 Semester Start Date: September 1st, 2025\n'
          '📌 Add/Drop Period: September 1st - September 15th\n'
          '📌 Mid-term Exams: October 20th - October 27th\n'
          '📌 Final Exams: December 10th - December 20th\n'
          '📌 Semester End Date: December 20th, 2025\n\n'
          '💡 Mark your calendar! Late registration incurs a £50 fee.']),

        (r'contact information|contact info|contacts',
         ['📞 CONTACT INFORMATION:\n'
          '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
          '👤 Academic Advisor:\n'
          '   Dr. Jane Smith\n'
          '   📧 jane.smith@example.com\n'
          '   ☎️ +44 123 456 7890\n'
          '   🏢 Office: Building A, Room 205\n'
          '   🕐 Office Hours: Mon-Fri, 10:00 AM - 4:00 PM\n\n'
          '🏫 Program Coordinator:\n'
          '   Prof. John Doe\n'
          '   📧 john.doe@example.com\n'
          '   ☎️ +44 123 456 7891\n\n'
          '📚 Student Services:\n'
          '   📧 studentservices@example.com\n'
          '   ☎️ +44 123 456 7892\n\n'
          '💰 Finance Office:\n'
          '   📧 finance@example.com\n'
          '   ☎️ +44 123 456 7893']),

        # SPECIFIC COURSE QUESTIONS
        (r'What courses are offered|courses offered|what courses|list courses',
         ['The MSc program offers courses in Advanced Algorithms, Machine Learning, Cybersecurity, Data Science, Artificial Intelligence, Natural Language Processing, Computer Vision, and Big Data Analytics.']),

        (r'How can I enroll|how to enroll|enroll in course|register for course',
         ['You can enroll in a course by visiting the course enrollment page on the university website at www.university.edu/enrollment. The process takes about 10 minutes.']),

        # CONTACT QUESTIONS
        (r'Who should I contact|academic advisor|advising contact|who to contact',
         ['You can contact the academic advisor, Dr. Jane Smith, at jane.smith@example.com or call +44 123 456 7890. Her office is in Building A, Room 205.']),

        # DATE QUESTIONS
        (r'When does the semester start|semester start|start date',
         ['The semester starts on September 1st, 2025.']),

        (r'What is the deadline|registration deadline|deadline for registration',
         ['The deadline for course registration is August 25th, 2025. Late registration incurs a £50 fee.']),

        # GREETINGS
        (r'Hi|Hello|Hey|Greetings|Good morning|Good afternoon|Good evening',
         ['Hello! Welcome to the MSc Program FAQ Bot. How can I help you today?',
          'Hi there! I\'m here to answer your questions about the MSc program.',
          'Hey! Ask me anything about courses, enrollment, or program details.']),

        # HELP COMMAND
        (r'Help|help|what can you do|menu|options',
         ['You can ask me about:\\n\\n'
          '📚 1. Course information - Type: "course information"\\n'
          '📝 2. Enrollment procedures - Type: "Course enrollment procedures"\\n'
          '📅 3. Important dates - Type: "important dates"\\n'
          '📞 4. Contact information - Type: "contact information"\\n\\n'
          'Or ask specific questions like:\\n'
          '• "What courses are offered?"\\n'
          '• "How can I enroll?"\\n'
          '• "When does the semester start?"\\n'
          '• "Who should I contact for advising?"']),

        # THANKS
        (r'thank you|thanks|thank|appreciate',
         ['You\'re welcome! Happy to help!',
          'Glad I could assist you!',
          'Anytime! Feel free to ask more questions.']),

        # EXIT COMMANDS
        (r'exit|quit|bye|goodbye|see you',
         ['Goodbye! Have a great day! Feel free to come back anytime.',
          'Thanks for chatting! Good luck with your studies!',
          'See you later! Don\'t hesitate to return if you have more questions.']),

        # CATCH-ALL FOR UNKNOWN QUESTIONS
        (r'(.*)',
         ["I'm sorry, I don't understand that question. Type 'Help' to see what I can answer, or try one of these:\\n"
          "• 'course information'\\n"
          "• 'enrollment procedures'\\n"
          "• 'important dates'\\n"
          "• 'contact information'"])
    ]

    return Chat(pairs, reflections)


def main():
    # Download NLTK data
    download_nltk_data()

    # Initialize chatbot
    chatbot = setup_chatbot()

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

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.respond(prompt)

                # Handle empty responses
                if not response or response == "None":
                    response = "I'm sorry, I don't understand that question. Type 'Help' to see what I can answer."

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
