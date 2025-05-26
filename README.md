# RAG Chatbot for MTech Applied AI Course

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions about the MTech Applied AI course at VNIT Nagpur. It uses LangChain, FAISS, and Streamlit to provide accurate and context-aware responses about the course curriculum, faculty, schedule, grading, and more.

## Features
- PDF and TXT document ingestion with semantic chunking
- Vector search using FAISS with HuggingFace embeddings
- Question answering using Zephyr LLM
- Smart follow-up question generation
- Beautiful Streamlit interface with chat bubbles
- Sidebar with course structure navigation
- Context-aware responses without document references
- Strict adherence to curriculum content

## Document Support
- curriculum.txt: Course structure, subjects, and descriptions
- Schedule.pdf: Class timings and faculty information
- Faculty.pdf: Professor details and roles
- Grading.pdf: Grade categories and evaluation methods
- admission.pdf: Admission process and requirements

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

3. Add your HuggingFace API key to `.env`:
```env
HUGGINGFACE_API_KEY=""
```

4. Add your course documents to the `data/` folder:
   - curriculum.txt
   - Schedule.pdf
   - Faculty.pdf
   - Grading.pdf
   - admission.pdf

5. Run the app:
```bash
streamlit run app/main.py
```

## Project Structure
```
rag_course_chatbot/
├── app/
│   └── main.py          # Main application code
├── data/
│   ├── curriculum.txt   # Course structure and content
│   ├── Schedule.pdf     # Class schedules
│   ├── Faculty.pdf      # Faculty information
│   ├── Grading.pdf      # Grading system
│   └── admission.pdf    # Admission process
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Features in Detail

### Document Processing
- Semantic chunking for curriculum.txt
- Custom chunking for PDF documents
- Metadata preservation for better context

### Question Answering
- Context-aware responses
- No document references in answers
- Complete information from curriculum
- Strict adherence to available content

### Follow-up Questions
- Smart question generation
- Topic-based question selection
- Prevention of repeated questions
- Context-aware suggestions

### User Interface
- Modern chat interface
- Course structure in sidebar
- Real-time responses
- Beautiful styling

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
