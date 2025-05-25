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

# --- Sidebar: Branding, Info, and Navigation ---
with st.sidebar:
    st.image("data/vnit_logo.png", width=140)
    st.title("MTech Applied AI Chatbot")
    st.markdown("""
    **Program Knowledge Assistant**  
    - Ask about syllabus, faculty, schedule, grading, and more.
    - Powered by RAG, Zephyr LLM, and semantic chunking.
    """)
    st.markdown("---")
    st.markdown("**Contact:** support@vnit.ac.in")
    st.markdown("**Version:** 1.0.0")

# --- Prompt Template ---
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "<|system|>\n"
        "You are a helpful assistant for answering questions about the MTech Applied AI course. "
        "Use ONLY the following context to answer. If unsure, say you don't know.\n"
        "</s>\n"
        "<|user|>\n"
        "Context:\n{context}\n\nQuestion: {question}\n"
        "</s>\n"
        "<|assistant|>"
    ),
)

@st.cache_resource
def build_vector_store():
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if not pdf_files:
        st.error("No PDF files found in the data directory")
        st.stop()

    st.info(f"Loading {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    custom_chunks = []

    for pdf_file in pdf_files:
        if pdf_file == "Schedule.pdf":
            custom_chunks.extend([
                Document(page_content="The schedule outlines online session timings, faculty names, and WebEx links for AI, Machine Learning, and Data Science courses...", metadata={"source": "Schedule.pdf", "title": "Overview"}),
                Document(page_content="Saturday Regular Classes: AI in Healthcare — Dr. Vipin Kamble, 2:00–5:00 PM; Deep Learning Techniques — Dr. Anamika Singh...", metadata={"source": "Schedule.pdf", "title": "Regular Sessions - Saturday"}),
                Document(page_content="Sunday Regular Classes: NLP — Dr. Saugata Sinha, 2:00–5:00 PM; Big Data Analytics — Ashish Tiwari, 11:30–2:30 PM", metadata={"source": "Schedule.pdf", "title": "Regular Sessions - Sunday"}),
                Document(page_content="Saturday Regular & Backlogs: Data Transformation — Dr. Saugata Sinha, 8:00–11:00 AM; Computer Vision — Dr. Vishal Satpute...", metadata={"source": "Schedule.pdf", "title": "Regular & Backlog - Saturday"}),
                Document(page_content="Sunday Regular & Backlogs: Statistics for ML (also known as Statistics) — Dr. Prabhat Sharma; Programming for Data Science — Dr. Praveen Pawar", metadata={"source": "Schedule.pdf", "title": "Regular & Backlog - Sunday"}),
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
        elif pdf_file == "curriculum.pdf":
            custom_chunks.extend([
                Document(
                    page_content="""The subjects covered in the first semester of the M.Tech Applied AI program are:
1. Programming for Data Science – covering Python, object-oriented programming, debugging, and data visualization using libraries like Pandas and Matplotlib.
2. Statistics for Machine Learning – includes probability theory, statistical inference, optimization techniques, and linear algebra.
3. Computer Vision – introduces image processing, feature extraction (e.g., SIFT, HOG), and object detection techniques.
4. Data Transformation – focuses on SQL, database normalization (1NF to BCNF), and advanced NoSQL querying.
""",
                    metadata={"source": "curriculum.pdf", "title": "Semester 1"}
                ),
                Document(
                    page_content="""The subjects in the second semester of the M.Tech Applied AI program are:
1. Neural Networks – covers backpropagation, CNNs, RNNs, transformers, and deep reinforcement learning.
2. Machine Learning Algorithms and Applications – explores decision trees, SVMs, clustering, Bayesian networks, and model evaluation.
3. Elective 1 – chosen from a specialized list of AI topics.
4. Elective 2 – another elective tailored to the student’s area of interest.
5. Mini Project – hands-on project involving model training and deployment.
""",
                    metadata={"source": "curriculum.pdf", "title": "Semester 2"}
                ),
                Document(
                    page_content="""The third semester consists of:
1. Elective 3 – advanced topic in AI such as NLP or Big Data.
2. Elective 4 – further specialization course.
3. Dissertation Phase I – research problem formulation, literature survey, and initial experiments.
""",
                    metadata={"source": "curriculum.pdf", "title": "Semester 3"}
                ),
                Document(
                    page_content="""The fourth semester includes:
1. Dissertation Phase II – final research implementation, evaluation, and thesis writing.
2. Personality Development and Communication Skills – an audit course focused on soft skills and professional communication.
""",
                    metadata={"source": "curriculum.pdf", "title": "Semester 4"}
                ),
                Document(
                    page_content="""Elective Subjects in the program include:
1. Internet of Things and Embedded Systems – sensor networks, BLE/Wi-Fi communication, and IoT platforms.
2. Deep Learning Techniques – CNNs, YOLO, GANs, transformers, and reinforcement learning.
3. Natural Language Processing – text classification, sentiment analysis, transformers (BERT, GPT).
4. Big Data Analytics – Hadoop, Spark, real-time processing with Kafka and Flink.
5. Deployment of ML Models – MLOps practices including CI/CD, model serving, and cloud deployment.
6. AI in Healthcare – applications in diagnostics, EHR analysis, drug discovery.
7. Applied Signal Processing – DSP fundamentals, time-frequency analysis, AI in signal processing.
8. AI Workshop – hands-on training with end-to-end machine learning pipelines.
""",
                    metadata={"source": "curriculum.pdf", "title": "Elective Subjects"}
                ),
                Document(
                    page_content="""Course Description for Deep Learning Techniques:
This elective covers:
- CNN architectures: VGG, ResNet, EfficientNet
- Object detection: YOLO, Faster R-CNN
- Generative models: GANs, autoencoders
- Sequence models: transformers, attention mechanisms
- Reinforcement learning: Q-learning, policy gradients
""",
                    metadata={"source": "curriculum.pdf", "title": "Deep Learning Techniques"}
                )
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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(custom_chunks, embeddings)
    return db

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

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

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

    # Display chat history with enhanced UI
    for message in st.session_state.chat_history:
        render_message(message["role"], message["content"], message.get("timestamp"))

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

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["result"],
            "timestamp": datetime.now().strftime("%H:%M")
        })
        render_message("assistant", response["result"])

        # Optionally, show source documents
        if response.get("source_documents"):
            with st.expander("Source Documents"):
                for doc in response["source_documents"]:
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(doc.page_content)
                    st.markdown("---")

if __name__ == "__main__":
    main()