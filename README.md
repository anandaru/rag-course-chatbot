# RAG Chatbot for MTK Applied AI Course

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions about the MTK Applied AI course, including syllabus, subjects, instructors, and schedule. It uses LangChain, FAISS, and Streamlit.

## Features
- PDF ingestion and chunking
- Vector search with FAISS
- Question answering using OpenAI LLM
- Follow-up question suggestions
- Streamlit interface

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_course_chatbot.git
cd rag_course_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key to `.env`:
```env
OPENAI_API_KEY=your_key_here
```

4. Add your course PDF to the `data/` folder.

5. Run the app:
```bash
streamlit run app/main.py
```
