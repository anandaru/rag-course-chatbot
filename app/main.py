import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Load environment variables ---
load_dotenv()

# --- Custom CSS for enterprise look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stChatMessage {
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        max-width: 80%;
        font-size: 1.1em;
    }
    .user-bubble {
        background: linear-gradient(90deg, #0052cc 0%, #007fff 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .assistant-bubble {
        background: #fff;
        color: #222;
        border: 1px solid #e3e8ee;
        margin-right: auto;
        text-align: left;
    }
    .timestamp {
        font-size: 0.8em;
        color: #888;
        margin-top: 2px;
    }
    .st-emotion-cache-1avcm0n {
        padding-top: 0rem;
    }
    .st-emotion-cache-1v0mbdj {
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_vector_store():
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    txt_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    if not pdf_files and not txt_files:
        st.error("No PDF or TXT files found in the data directory")
        st.stop()

    st.info(f"Loading {len(pdf_files)} PDF files and {len(txt_files)} TXT files")
    custom_chunks = []

    # Process curriculum.txt first
    if "curriculum.txt" in txt_files:
        st.info("Processing curriculum.txt")
        with open(os.path.join("data", "curriculum.txt"), 'r') as file:
            raw_text = file.read()

        def split_curriculum_into_main_sections(text):
            # Define the main sections we want to extract
            main_sections = [
                "1. Program Overview",
                "2. Semester-Wise Breakdown of credits",
                "3. Core Course Descriptions with Textbooks , reference books and Key Topics",
                "4. Elective Course Descriptions with Key Topics",
                "5. Credit Summary"
            ]
            
            chunks = []
            current_section = None
            current_content = []
            
            # Split text into lines
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line starts a new main section
                is_main_section = False
                for section in main_sections:
                    if line.startswith(section):
                        # If we were building a previous section, save it
                        if current_section and current_content:
                            chunks.append(Document(
                                page_content='\n'.join(current_content),
                                metadata={"source": "curriculum.txt", "title": current_section}
                            ))
                        current_section = section
                        current_content = [line]
                        is_main_section = True
                        break
                
                # If not a main section, add to current content
                if not is_main_section and current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section and current_content:
                chunks.append(Document(
                    page_content='\n'.join(current_content),
                    metadata={"source": "curriculum.txt", "title": current_section}
                ))
            
            return chunks

        curriculum_chunks = split_curriculum_into_main_sections(raw_text)
        st.info(f"Created {len(curriculum_chunks)} main chunks from curriculum.txt")
        custom_chunks.extend(curriculum_chunks)

    # Process PDF files
    for pdf_file in pdf_files:
        if pdf_file == "Schedule.pdf":
            custom_chunks.extend([
                Document(page_content="The schedule outlines online session timings, faculty names, and WebEx links for AI, Machine Learning, and Data Science courses...", metadata={"source": "Schedule.pdf", "title": "Overview"}),
                Document(page_content="Saturday Regular Classes: AI in Healthcare  is taught by Dr. Vipin Kamble, 2:00–5:00 PM; Deep Learning Techniques  is taught by Dr. Anamika Singh...", metadata={"source": "Schedule.pdf", "title": "Regular Sessions - Saturday"}),
                Document(page_content="Sunday Regular Classes: NLP   is taught by Dr. Saugata Sinha, 2:00–5:00 PM; Big Data Analytics  is taught by Ashish Tiwari, 11:30–2:30 PM", metadata={"source": "Schedule.pdf", "title": "Regular Sessions - Sunday"}),
                Document(page_content="Saturday Regular & Backlogs: Data Transformation  is taught by Dr. Saugata Sinha, 8:00–11:00 AM; Computer Vision  is taught by Dr. Vishal Satpute...", metadata={"source": "Schedule.pdf", "title": "Regular & Backlog - Saturday"}),
                Document(page_content="Sunday Regular & Backlogs: Statistics for ML (also known as Statistics)  is taught by Dr. Prabhat Sharma; Programming for Data Science  is taught by Dr. Praveen Pawar", metadata={"source": "Schedule.pdf", "title": "Regular & Backlog - Sunday"}),
                Document(page_content="Additional Backlog: Neural Networks, IoT, Deployment of ML Models", metadata={"source": "Schedule.pdf", "title": "Additional Backlog"}),
                Document(page_content="Key takeaways: Regular sessions 2–5 PM; Backlogs in AM slots; Industry experts contribute", metadata={"source": "Schedule.pdf", "title": "Key Takeaways"})
            ])
        elif pdf_file == "admission.pdf":
            custom_chunks.extend([
                Document(page_content="Step-by-step admission process including self-registration, fee payment, and institute reporting.", metadata={"source": "admission.pdf", "title": "Admission Steps"}),
                Document(page_content="List of required documents for VNIT admission: seat allotment letter, photo ID, degree certificates, etc.", metadata={"source": "admission.pdf", "title": "Required Documents"}),
                Document(page_content="Instructions for online fee payment using VNIT Pay and steps to complete registration.", metadata={"source": "admission.pdf", "title": "Fee Payment and Final Steps"}),
                Document(page_content="Details about cancellation and refund policy with VNIT link.", metadata={"source": "admission.pdf", "title": "Cancellation Policy"})
            ])
        elif pdf_file == "Faculty.pdf":
            custom_chunks.extend([
                Document(page_content="Accreditations: Center of Excellence in Embedded Systems, NBA accreditation for B.Tech and M.Tech programs.", metadata={"source": "Faculty.pdf", "title": "Accreditations & Recognitions"}),
                Document(page_content="Research Areas: Communication, Image & Signal Processing, Embedded Systems, RF, Antenna Design.", metadata={"source": "Faculty.pdf", "title": "Research Focus Areas"}),
                Document(page_content="Professors with contact details: Dr. Abhay Gandhi, Dr. Bhurchandi, Dr. Keskar, Dr. Ashwin Kothari.", metadata={"source": "Faculty.pdf", "title": "Faculty and Staff - Professors"}),
                Document(page_content="Associate Professor: Dr. Vishal Satpute with contact details and specialization.", metadata={"source": "Faculty.pdf", "title": "Faculty and Staff - Associate Professors"}),
                Document(page_content="Assistant Professors with emails and phones: Dr. Ankit Bhurane, Dr. Saugata Sinha, Dr. Praveen Pawar, etc.", metadata={"source": "Faculty.pdf", "title": "Faculty and Staff - Assistant Professors"}),
                Document(page_content="Technical and Administrative staff roles and contact details for labs and operations.", metadata={"source": "Faculty.pdf", "title": "Technical & Administrative Staff"})
            ])
        elif pdf_file == "Grading.pdf":
            custom_chunks.extend([
                Document(page_content="Grade Categories: AA (10) to FF (0), including audit grades NP/NF and special categories SS/ZZ.", metadata={"source": "Grading.pdf", "title": "Grade Categories"}),
                Document(page_content="Implications of FF Grade and re-attempt policy for core and elective courses.", metadata={"source": "Grading.pdf", "title": "FF Grade Policy"}),
                Document(page_content="SGPA and CGPA Calculation methods, formula, and course weightage.", metadata={"source": "Grading.pdf", "title": "CGPA Calculation"}),
                Document(page_content="Course Evaluation methods for lectures, practicals, and projects.", metadata={"source": "Grading.pdf", "title": "Course Evaluation"})
            ])
        else:
            loader = PyMuPDFLoader(os.path.join("data", pdf_file))
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            custom_chunks.extend(text_splitter.split_documents(docs))

    st.info(f"Total chunks created: {len(custom_chunks)}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(custom_chunks, embeddings)
    return db

# --- Prompt Template ---
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "<|system|>\n"
        "You are a helpful assistant for answering questions about the MTech Applied AI course. "
        "Use the following context to answer. The context comes from multiple documents. "
        "If the information is available in the context, provide a complete answer. "
        "If unsure, say you don't know.\n\n"
        "Important rules:\n"
        "1. DO NOT mention or reference any source documents (PDFs, TXT files, etc.) in your answers\n"
        "2. DO NOT say things like 'according to the document' or 'as mentioned in the PDF'\n"
        "3. DO NOT mention where the information comes from\n"
        "4. Just provide the information directly as if it's your knowledge\n"
        "5. If information is not available in the context, simply say 'I don't have that information'\n"
        "6. For course structure questions, ALWAYS include:\n"
        "   - All subjects listed in the semester\n"
        "   - Credit hours for each subject\n"
        "   - Total credits for the semester\n"
        "7. For subject-related questions, include:\n"
        "   - All topics covered\n"
        "   - Textbooks and reference books\n"
        "   - Key learning outcomes\n"
        "8. IMPORTANT: DO NOT make assumptions or add information not present in the context\n"
        "9. For semester-wise questions, ONLY list the subjects and credits exactly as shown in the curriculum\n"
        "10. DO NOT describe subjects or topics unless that information is explicitly provided in the context\n"
        "</s>\n"
        "<|user|>\n"
        "Context:\n{context}\n\nQuestion: {question}\n"
        "</s>\n"
        "<|assistant|>"
    ),
)

def render_message(role, content, timestamp=None):
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    avatar = "🧑" if role == "user" else "🤖"
    if not timestamp:
        timestamp = datetime.now().strftime("%H:%M")
    st.markdown(
        f"""
        <div class="stChatMessage {bubble_class}">
            <div style="display: flex; align-items: center;">
                <span style="font-size:1.5em; margin-right: 8px;">{avatar}</span>
                <div>
                    <div>{content}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("MTech Applied AI Course Chatbot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "follow_up_questions" not in st.session_state:
        st.session_state.follow_up_questions = []
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = build_vector_store()
    if "previous_questions" not in st.session_state:
        st.session_state.previous_questions = set()
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    # Display curriculum chunks in sidebar
    with st.sidebar:
        st.image("data/vnit_logo.png", width=140)
        st.title("MTech Applied AI Course Chatbot")
        st.markdown("""
        **Program Knowledge Assistant**  
        - Ask about syllabus, faculty, schedule, grading, and more.
        - Powered by RAG and Zephyr LLM.
        """)
        st.markdown("---")
        
        # Display curriculum chunks
        st.markdown("**Course Structure:**")
        
        # Debug information
        all_docs = list(st.session_state.vector_store.docstore._dict.values())
        st.write(f"Total documents: {len(all_docs)}")
        
        # Filter curriculum chunks
        curriculum_chunks = [doc for doc in all_docs 
                           if doc.metadata.get("source") == "curriculum.txt"]
        st.write(f"Curriculum chunks: {len(curriculum_chunks)}")
        
        # Sort chunks by their title (which contains the section number)
        curriculum_chunks.sort(key=lambda x: x.metadata.get("title", ""))
        
        # Display each chunk
        for chunk in curriculum_chunks:
            title = chunk.metadata.get("title", "Section")
            content = chunk.page_content
            with st.expander(title):
                st.write(content)
                st.markdown(f"*Source: {chunk.metadata.get('source', '')}*")
        
        st.markdown("---")
        st.markdown("**Contact:** support@vnit.ac.in")
        st.markdown("**Version:** 1.0.0")

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 7})

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 512,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    def is_similar_question(new_q, existing_q):
        # First check for exact match (case-insensitive)
        if new_q.lower() == existing_q.lower():
            return True
            
        # Convert to lowercase and remove punctuation for comparison
        new_q = ''.join(c.lower() for c in new_q if c.isalnum() or c.isspace())
        existing_q = ''.join(c.lower() for c in existing_q if c.isalnum() or c.isspace())
        
        # Check if one is a substring of the other
        if new_q in existing_q or existing_q in new_q:
            return True
        
        # Check for high word overlap
        new_words = set(new_q.split())
        existing_words = set(existing_q.split())
        overlap = len(new_words.intersection(existing_words))
        return overlap > len(new_words) * 0.6  # Lowered threshold to 60% for stricter matching

    def generate_unique_question(context, answer):
        # List of default questions that are always relevant
        default_questions = [
            "What are the core subjects in the first semester?",
            "Who are the faculty members teaching the program?",
            "What are the class timings for regular sessions?",
            "What is the project evaluation process?",
            "What are the elective subjects available?",
            "What is the dissertation process?",
            "What are the attendance requirements?",
            "What is the fee structure?",
            "What are the lab requirements?",
            "What is the semester examination pattern?",
            "What are the project submission deadlines?",
            "What is the minimum attendance requirement?",
            "What are the course prerequisites?",
            "What is the course duration?",
            "What are the specialization options?",
            "What is the internship process?",
            "What are the research opportunities?",
            "What is the placement process?",
            "What are the library facilities?",
            "What is the grading system?",
            "What are the evaluation criteria?",
            "What are the course outcomes?",
            "What are the learning objectives?",
            "What are the assessment methods?"
        ]

        # Try to generate a new question
        follow_up_prompt = f"""Based on the following answer and context, generate ONE follow-up question about the MTech Applied AI course.
        Rules:
        1. The question MUST start with a question word (What, When, Where, Who, How)
        2. The question MUST ONLY be about information that is EXPLICITLY mentioned in the provided context
        3. DO NOT ask about:
           - Information not present in the context
           - Background details not mentioned
           - Expertise not explicitly stated
           - General AI/ML concepts
           - Technical implementations
           - Real-world applications
           - Research topics
        4. The question must end with a question mark
        5. IMPORTANT: Only use facts that are directly stated in the context
        6. DO NOT repeat or ask similar questions to these previously asked questions: {list(st.session_state.previous_questions)}
        7. DO NOT repeat or ask similar questions to the last question: {st.session_state.last_question}
        8. IMPORTANT: The question MUST be different from all previous questions
        9. DO NOT ask about the same topic as the current question
        10. DO NOT ask about documents, requirements, or procedures that were just discussed
        
        Current question: {st.session_state.last_question}
        Current answer: {answer}
        
        Context: {context}
        
        Follow-up question:"""

        follow_up_response = llm(follow_up_prompt)
        # Clean and validate question
        question = None
        for line in follow_up_response.split('\n'):
            line = line.strip()
            if line and line[0].isupper() and line.endswith('?'):
                question = line
                break

        # Check if the question is similar to any previous questions
        if question:
            # First check for exact match
            if question in st.session_state.previous_questions or question == st.session_state.last_question:
                question = None
            else:
                # Then check for similarity
                for prev_q in st.session_state.previous_questions:
                    if is_similar_question(question, prev_q):
                        question = None
                        break
                if question and st.session_state.last_question and is_similar_question(question, st.session_state.last_question):
                    question = None

        # If no valid question or question was similar to previous ones, use an unused default question
        if not question:
            for q in default_questions:
                if q not in st.session_state.previous_questions and q != st.session_state.last_question:
                    is_unique = True
                    for prev_q in st.session_state.previous_questions:
                        if is_similar_question(q, prev_q):
                            is_unique = False
                            break
                    if is_unique and (not st.session_state.last_question or not is_similar_question(q, st.session_state.last_question)):
                        question = q
                        break

        # If all questions have been used, clear the history and start fresh
        if not question:
            st.session_state.previous_questions.clear()
            st.session_state.question_count = 0
            question = default_questions[0]

        # Update tracking
        st.session_state.previous_questions.add(question)
        st.session_state.last_question = question
        st.session_state.question_count += 1
        return question

    # Display chat history with enhanced UI
    for message in st.session_state.chat_history:
        render_message(message["role"], message["content"], message.get("timestamp"))

    # Handle follow-up question selection
    if st.session_state.selected_question:
        question = st.session_state.selected_question
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        render_message("user", question)

        with st.spinner("Thinking..."):
            response = qa_chain({"query": question})
            new_question = generate_unique_question(response.get('source_documents', []), response['result'])
            st.session_state.follow_up_questions = [new_question]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["result"],
            "timestamp": datetime.now().strftime("%H:%M")
        })
        render_message("assistant", response["result"])
        st.session_state.selected_question = None

    # Chat input
    if prompt := st.chat_input("What would you like to know about the course?"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        render_message("user", prompt)

        with st.spinner("Thinking..."):
            response = qa_chain({"query": prompt})
            new_question = generate_unique_question(response.get('source_documents', []), response['result'])
            st.session_state.follow_up_questions = [new_question]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["result"],
            "timestamp": datetime.now().strftime("%H:%M")
        })
        render_message("assistant", response["result"])

    # Display follow-up questions as clickable buttons
    if st.session_state.follow_up_questions:
        st.markdown("**Suggested follow-up questions:**")
        for idx, question in enumerate(st.session_state.follow_up_questions):
            if st.button(question, key=f"follow_up_{idx}"):
                st.session_state.selected_question = question
                st.rerun()

if __name__ == "__main__":
    main()