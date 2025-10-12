# 🩺 Medical Chat — AI-Powered Medical Assistant

An intelligent **medical chatbot** built using **LangChain**, **Groq (LLaMA 3.1)**, and **HuggingFace embeddings**, with **Streamlit** for a user-friendly interface.  
The app reads medical PDFs (such as the *Gale Encyclopedia of Medicine*), stores their embeddings in **ChromaDB**, and answers user queries **based only on the uploaded documents**.

---

## 🚀 Features

- 💬 Interactive medical Q&A through a clean Streamlit chat interface  
- 📚 Works with the *Gale Encyclopedia of Medicine* or **any PDF-based medical text**  
- 🧠 Uses **HuggingFace sentence-transformer embeddings** for semantic understanding  
- ⚡ Powered by **Groq’s LLaMA 3.1-8B** model for fast and accurate responses  
- 💾 Embeddings stored persistently in **Chroma vector database**  
- 🧍‍♀️ Context-aware and medically responsible responses  
- 🔒 Avoids hallucination — politely says “I don’t know” when context is insufficient  

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit |
| **LLM** | Groq LLaMA 3.1-8B (via LangChain) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | Chroma |
| **Document Loader** | LangChain PyPDFDirectoryLoader |
| **Environment** | Python 3.10+, dotenv |

---

## 📂 Project Structure

```
medical-chat/
│
├── data/                      # Folder containing PDFs (e.g. Gale Encyclopedia or any other medical book)
├── saved_embeddings/           # Persistent ChromaDB directory
├── app.py                      # Main Streamlit app file
├── .env                        # Environment variables (API keys etc.)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rehman047/Medical_ChatBot.git
cd Medical_ChatBot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate       # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:
```
langchain
langchain-community
langchain-core
langchain-huggingface
langchain-groq
langchain-chroma
streamlit
python-dotenv
chromadb
sentence-transformers
```

### 4. Add Environment Variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧠 Usage

### Step 1: Add a Medical PDF
Before running the app, **add at least one PDF file** to the `data/` directory.  
You can use the *Gale Encyclopedia of Medicine* or **any other medical textbook or reference guide**.  
The chatbot will use this data to answer questions contextually.

### Step 2: Run the App
```bash
streamlit run app.py
```

### Step 3: Chat
Ask any medical question — for example:  
> “What are the symptoms of diabetes?”  
> “How does insulin work?”  

The model will search your uploaded documents, extract relevant context, and answer responsibly.

---

## 🧰 Optional: Rebuild Embeddings

If you add or replace PDFs, uncomment the following lines in your code before running once:
```python
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=280)
chunks = splitter.split_documents(docs)
vector_store.add_documents(chunks)
```

This regenerates embeddings for all PDFs in the `data/` directory and stores them persistently.

---

## 🧑‍⚕️ Example Query

**User:** “What is hypertension?”  
**AI:**  
> Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is consistently high. It increases the risk of heart disease and stroke. Treatment often includes lifestyle changes such as diet, exercise, and sometimes medication.

---

## ⚠️ Disclaimer

> This chatbot is for **educational purposes only**.  
> It is **not a substitute for professional medical advice, diagnosis, or treatment**.  
> Always seek guidance from a qualified healthcare provider for medical concerns.

---

