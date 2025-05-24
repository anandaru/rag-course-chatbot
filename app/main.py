import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set HuggingFace API token


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load or build vector store
@st.cache_resource
def build_vector_store():
    try:
        # Get all PDF files from the data directory
        pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if not pdf_files:
            st.error("No PDF files found in the data directory")
            st.stop()
            
        st.info(f"Loading {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
        all_documents = []
        
        # Load each PDF
        for pdf_file in pdf_files:
            try:
                loader = PyMuPDFLoader(os.path.join("data", pdf_file))
                documents = loader.load()
                all_documents.extend(documents)
                st.success(f"Successfully loaded {pdf_file}")
            except Exception as e:
                st.error(f"Error loading {pdf_file}: {str(e)}")
                continue
        
        if not all_documents:
            st.error("No documents were loaded from the PDF files")
            st.stop()
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)
        st.info(f"Created {len(docs)} document chunks")
        
        # Create vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        st.success("Vector store created successfully")
        return db
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        st.stop()

# Custom QA prompt
qa_prompt = PromptTemplate(
    template="""<|system|>
You are an assistant for the MTK Applied AI course at VNIT.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
</s>

<|assistant|>
Let me help you with that. Based on the provided context, here's the answer:
</s>

<|user|>
Please provide a clear and concise answer.
</s>

<|assistant|>""",
    input_variables=["context", "question"]
)

# Follow-up question chain
def get_followup_chain():
    followup_prompt = PromptTemplate(
        input_variables=["question", "answer", "chat_history"],
        template="""<|system|>
You are an assistant for the MTK Applied AI course chatbot.
Based on the conversation history and the current Q&A, suggest 3 intelligent follow-up questions.

Chat History:
{chat_history}

Current Question: "{question}"
Current Answer: "{answer}"
</s>

<|assistant|>
I'll help you generate relevant follow-up questions. Here are 3 questions that would help you learn more about the topic:
</s>

<|user|>
Please format the questions as a bullet list, with each question on a new line starting with "- ".
</s>

<|assistant|>"""
    )
    return LLMChain(
        llm=HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 512,
                "top_p": 0.95,
                "repetition_penalty": 1.15
            }
        ),
        prompt=followup_prompt
    )

# Main UI
def main():
    st.title("MTK Applied AI Assistant")
    st.markdown("Ask anything about the course syllabus, schedule, subjects, or instructors.")

    try:
        # Initialize the vector store and chains
        with st.spinner("Loading documents and initializing AI..."):
            db = build_vector_store()
            retriever = db.as_retriever(search_kwargs={"k": 3})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFaceHub(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    model_kwargs={
                        "temperature": 0.1,  # Changed from 0 to 0.1
                        "max_new_tokens": 512,
                        "top_p": 0.95,
                        "repetition_penalty": 1.15
                    }
                ),
                retriever=retriever,
                chain_type_kwargs={"prompt": qa_prompt}
            )
            
            followup_chain = get_followup_chain()

        # Display chat history
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # Get user input
        user_question = st.chat_input("Ask your question:")

        if user_question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(user_question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = qa_chain.run(user_question)
                    st.write(answer)

                    # Get follow-up questions
                    chat_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-3:]])
                    followups = followup_chain.run({
                        "question": user_question,
                        "answer": answer,
                        "chat_history": chat_history_str
                    })

                    # Display follow-up questions as buttons
                    st.subheader("Suggested Follow-up Questions:")
                    followup_list = [q.strip()[2:] for q in followups.split("\n") if q.strip().startswith("- ")]
                    cols = st.columns(len(followup_list))
                    for col, question in zip(cols, followup_list):
                        if col.button(question, key=question):
                            st.session_state.chat_history.append((user_question, answer))
                            st.experimental_rerun()

            # Update chat history
            st.session_state.chat_history.append((user_question, answer))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

