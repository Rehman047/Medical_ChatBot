from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("Medical Chat")
if "history" not in st.session_state:
    st.session_state.history=[]
else:
    for message in st.session_state.history:
        with st.chat_message(message["messenger"]):
            st.write(message["content"])
# loader=PyPDFDirectoryLoader("data/")
# docs=loader.load()
# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=280
# )
#chunks=splitter.split_documents(docs)
model_name="sentence-transformers/all-MiniLM-L6-v2"
embeddings=HuggingFaceEmbeddings(model_name=model_name)
vector_store=Chroma(collection_name="any",embedding_function=embeddings,persist_directory="saved_embeddings/")
#vector_store.add_documents(chunks)
query=st.chat_input("How can I help you? ")
if query:
    st.session_state.history.append({"messenger":"user","content":query})
    with st.chat_message("user"):
        st.write(query)
    got_docs=vector_store.similarity_search(query,k=5)
    context=""
    for doc in got_docs:
        context+=doc.page_content+"\n"

    template=PromptTemplate(
        
    template = """
        You are a knowledgeable and trustworthy **medical assistant** helping users understand medical topics.
        Use only the information provided in the context below to answer.
        If the context does not contain enough information, politely say you don’t know and avoid guessing.

        Your response should be:
        - Clear, medically accurate, and easy to understand.
        - Written in a calm and professional tone.
        - Include explanations if useful (e.g. how a medicine works, precautions, side effects).

    ---
    🧠 **Query:** {query}

    📘 **Context:** 
    {context}

    ---
    Please provide your answer below:
    """,
        input_variables=["query","context"]
    )
    model=ChatGroq(model="llama-3.1-8b-instant")
    parser=StrOutputParser()
    chain=template|model|parser
    response=chain.invoke({"query":query,"context":context})
    st.session_state.history.append({"messenger":"ai","content":response})
    with st.chat_message("ai"):
        st.write(response)














