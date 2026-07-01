# 🩺 MedAssist AI — Medical RAG Chatbot with LangGraph

An intelligent medical assistant built using **LangGraph**, **LangChain**, **Groq LLMs**, **ChromaDB**, and **Streamlit**.

MedAssist AI allows users to ask medical questions and receive answers grounded in trusted medical documents. The application uses Retrieval-Augmented Generation (RAG) to search a medical knowledge base and generate context-aware responses while minimizing hallucinations.

---

# 🚀 Features

* 🩺 Medical question answering powered by Retrieval-Augmented Generation (RAG)
* 🧠 LangGraph workflow architecture for scalable AI applications
* 💬 Modern ChatGPT-style Streamlit interface
* 📚 Answers grounded in uploaded medical PDFs
* ⚡ Fast inference using Groq-hosted LLaMA models
* 🧬 Semantic retrieval using HuggingFace embeddings
* 💾 Persistent vector storage using ChromaDB
* 🗂 Multiple chat sessions with conversation history
* 🔄 Thread-based memory using LangGraph checkpointers
* 🔍 Context retrieval before every response
* 🛡 Reduced hallucinations through document-grounded answers
* 📱 Responsive and professional medical-themed UI

---

# 🏗 Architecture

The application follows a LangGraph workflow:

```text
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
Generate Answer
      │
      ▼
Return Response
```

### Graph Flow

```text
START
  │
  ▼
Retrieve Relevant Chunks
  │
  ▼
Generate Medical Response
  │
  ▼
END
```

---

# 🧩 Tech Stack

| Component           | Technology                             |
| ------------------- | -------------------------------------- |
| Frontend            | Streamlit                              |
| Workflow Engine     | LangGraph                              |
| LLM                 | Groq LLaMA 3.1 8B                      |
| Embeddings          | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database     | ChromaDB                               |
| Retrieval Framework | LangChain                              |
| Document Loader     | PyPDFDirectoryLoader                   |
| Memory              | LangGraph MemorySaver                  |
| Environment         | Python 3.10+                           |

---

# 📂 Project Structure

```text
medical-chatbot/
│
├── data/                     # Medical PDFs
├── saved_embeddings/         # Persistent ChromaDB storage
│
├── backend.py                # LangGraph workflow and RAG logic
├── frontend.py               # Streamlit frontend
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Rehman047/Medical_ChatBot.git
cd Medical_ChatBot
```

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Suggested dependencies:

```text
langgraph
langchain
langchain-community
langchain-core
langchain-huggingface
langchain-groq
langchain-chroma
streamlit
chromadb
sentence-transformers
python-dotenv
pypdf
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 📚 Preparing the Knowledge Base

Place one or more medical PDFs inside:

```text
data/
```

Examples:

* Gale Encyclopedia of Medicine
* Medical textbooks
* Clinical reference guides
* Healthcare manuals

The system uses these documents as its knowledge source.

---

# 🧠 Building the Vector Database

When adding documents for the first time, generate embeddings using:

```python
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=280
)

chunks = splitter.split_documents(docs)

vector_store.add_documents(chunks)
```

This process:

1. Loads PDFs
2. Splits them into chunks
3. Creates embeddings
4. Stores embeddings in ChromaDB

The embeddings remain available across application restarts.

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run frontend.py
```

Open the URL displayed by Streamlit in your browser.

---

# 💬 Example Questions

You can ask questions such as:

* What are the symptoms of diabetes?
* How does insulin work?
* What causes hypertension?
* What is asthma?
* What are common treatments for migraines?
* What are the side effects of aspirin?

---

# 🧑‍⚕️ Example Interaction

### User

```text
What is hypertension?
```

### Assistant

```text
Hypertension, commonly known as high blood pressure,
is a condition in which the pressure of blood against
the artery walls remains consistently elevated.

According to the provided medical context,
untreated hypertension may increase the risk of
heart disease, stroke, and kidney problems.

Treatment typically involves lifestyle modifications
and, when necessary, medication prescribed by a
healthcare professional.
```

---

# ✨ New Improvements

Compared to the previous version, MedAssist AI now includes:

* LangGraph-based architecture
* Thread-based chat memory
* Multiple conversation support
* Sidebar chat history
* Better frontend design
* Cleaner separation of frontend and backend
* Easier future integration of tools and agents
* More maintainable production-style codebase

---

# 🔮 Future Enhancements

Potential future upgrades:

* Streaming responses
* PDF upload from UI
* Source citations and page references
* Medical image analysis
* Voice input/output
* Multi-document collections
* Web search integration
* Agentic workflows using LangGraph
* User authentication

---

# ⚠️ Medical Disclaimer

This application is intended for educational and informational purposes only.

It does not provide medical diagnosis, treatment, or professional healthcare advice. Always consult a qualified healthcare professional regarding any medical condition or treatment decision.

The quality of responses depends on the documents available in the knowledge base.
