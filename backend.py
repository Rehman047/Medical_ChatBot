from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------
# Vector Store
# ---------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="medical_docs",
    embedding_function=embeddings,
    persist_directory="saved_embeddings/"
)

# ---------------------------
# LLM
# ---------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# ---------------------------
# Prompt
# ---------------------------

prompt_template = PromptTemplate(
    template="""
You are a knowledgeable and trustworthy medical assistant.

Use ONLY the provided context.

If the answer cannot be found in the context,
say that you don't know.

Query:
{query}

Context:
{context}

Answer:
""",
    input_variables=["query", "context"]
)

# ---------------------------
# State
# ---------------------------

class MedicalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


# ---------------------------
# Nodes
# ---------------------------

def retrieve_documents(state: MedicalState):

    query = state["messages"][-1].content

    docs = vector_store.similarity_search(
        query,
        k=5
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


def generate_answer(state: MedicalState):

    query = state["messages"][-1].content

    prompt = prompt_template.format(
        query=query,
        context=state["context"]
    )

    response = llm.invoke(prompt)

    return {
        "messages": [response]
    }


# ---------------------------
# Graph
# ---------------------------

graph = StateGraph(MedicalState)

graph.add_node(
    "retrieve",
    retrieve_documents
)

graph.add_node(
    "generate",
    generate_answer
)

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

memory = MemorySaver()

chatbot = graph.compile(
    checkpointer=memory
)
