from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

MAX_RETRIES = 10
MAX_REWRITE_TRIES = 3


# ---------------------------
# Ingestion and retrieval
# ---------------------------

loader = PyPDFLoader("./docs/book.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)
chunks = splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="saved_embeddings/",
)

if len(vector_store) == 0:
    vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ---------------------------
# Model and structured outputs
# ---------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


class RetrieveDecision(BaseModel):
    need_retrieval: bool = Field(
        description="Whether the question requires information from the internal documents."
    )


class RelevanceDecision(BaseModel):
    relevant: bool = Field(
        description="Whether the document is on the same topic as the question."
    )


class SupportDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: str = Field(description="Evidence from the context supporting the grade.")


class UsefulnessDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    use_reason: str = Field(description="Why the answer does or does not answer the question.")


retrieve_decider = llm.with_structured_output(RetrieveDecision)
relevance_decider = llm.with_structured_output(RelevanceDecision)
support_decider = llm.with_structured_output(SupportDecision)
usefulness_decider = llm.with_structured_output(UsefulnessDecision)


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    answer: str
    issup: str
    evidence: str
    isuse: str
    use_reason: str
    retries: int


def _question(state: State) -> str:
    if state.get("question"):
        return state["question"]
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("A question is required.")
    content = messages[-1].content
    if not isinstance(content, str):
        raise ValueError("The latest message must contain text.")
    return content


def _answer_message(answer: str) -> AIMessage:
    return AIMessage(content=answer)


# ---------------------------
# Nodes
# ---------------------------

def decide_retrieval(state: State) -> dict:
    question = next(
        (
            message.content
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ),
        state.get("question"),
    )
    if not question:
        raise ValueError("A question is required.")
    decision = retrieve_decider.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Decide whether this question needs the internal PDF documents. "
                    "Prefer true when uncertain. Questions requiring current or "
                    "document-specific information need retrieval.",
                ),
                ("human", "{question}"),
            ]
        ).format_messages(question=question)
    )
    return {
        "question": question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "retries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "need_retrieval": decision.need_retrieval,
    }


def retrieve(state: State) -> dict:
    query = state.get("retrieval_query") or _question(state)
    docs = retriever.invoke(query)
    return {"docs": docs}


def is_relevant(state: State) -> dict:
    question = _question(state)
    relevant_docs: list[Document] = []
    for doc in state.get("docs", []):
        decision = relevance_decider.invoke(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Judge topic-level relevance only. Keep documents that discuss "
                        "the same subject as the question, even if they do not fully "
                        "answer it. Do not judge factual support here.",
                    ),
                    ("human", "Question: {question}\n\nDocument:\n{document}"),
                ]
            ).format_messages(question=question, document=doc.page_content)
        )
        if decision.relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def generate_from_context(state: State) -> dict:
    context = "\n---\n".join(doc.page_content for doc in state.get("relevant_docs", []))
    response = llm.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question using only the supplied context. Do not mention "
                    "the context, documents, retrieval, or these instructions. If the "
                    "context does not contain the answer, say so plainly.",
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        ).format_messages(question=_question(state), context=context)
    )
    answer = response.content
    return {"context": context, "answer": answer, "messages": [_answer_message(answer)]}


def generate_direct(state: State) -> dict:
    response = llm.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question from general knowledge. Be accurate and concise. "
                    "Do not claim to have consulted internal documents.",
                ),
                ("human", "{question}"),
            ]
        ).format_messages(question=_question(state))
    )
    answer = response.content
    return {"answer": answer, "messages": [_answer_message(answer)]}


def no_answer_found(state: State) -> dict:
    answer = "I could not find relevant information in the available documents."
    return {"answer": answer, "messages": [_answer_message(answer)]}


def is_sup(state: State) -> dict:
    decision = support_decider.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Grade whether the answer is supported by the context. Use "
                    "fully_supported only when every claim is supported. Downgrade to "
                    "partially_supported if the answer contains any unsupported "
                    "qualitative language (for example, calling something robust or "
                    "generous), even when its factual claims are otherwise correct.",
                ),
                ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
            ]
        ).format_messages(context=state.get("context", ""), answer=state.get("answer", ""))
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


def revise_answer(state: State) -> dict:
    response = llm.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the answer using only direct quotes from the context. Return "
                    "a bare bulleted list of quotes, with no connecting prose and no "
                    "unsupported wording.",
                ),
                ("human", "Context:\n{context}\n\nAnswer to revise:\n{answer}"),
            ]
        ).format_messages(
            context=state.get("context", ""),
            answer=state.get("answer", ""),
        )
    )
    answer = response.content
    return {
        "answer": answer,
        "retries": state.get("retries", 0) + 1,
        "messages": [_answer_message(answer)],
    }


def is_use(state: State) -> dict:
    decision = usefulness_decider.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Decide only whether the answer actually answers the question. "
                    "Do not re-check whether it is supported by the context.",
                ),
                ("human", "Question: {question}\n\nAnswer: {answer}"),
            ]
        ).format_messages(
            question=_question(state),
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.use_reason}


def rewrite_question(state: State) -> dict:
    response = llm.invoke(
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the question as a short, keyword-dense retrieval query "
                    "optimized for searching the internal PDFs. Return only the query.",
                ),
                ("human", "{question}"),
            ]
        ).format_messages(question=_question(state))
    )
    return {
        "retrieval_query": response.content.strip(),
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


# ---------------------------
# Routing and graph
# ---------------------------

def route_after_decide(state: State) -> str:
    return "retrieve" if state.get("need_retrieval", True) else "generate_direct"


def route_after_relevance(state: State) -> str:
    return "generate_from_context" if state.get("relevant_docs") else "no_answer_found"


def route_after_issup(state: State) -> str:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


def route_after_isuse(state: State) -> str:
    if state.get("isuse") == "useful":
        return "end"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


graph = StateGraph(State)
graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("generate_direct", generate_direct)
graph.add_node("retrieve", retrieve)
graph.add_node("is_relevant", is_relevant)
graph.add_node("generate_from_context", generate_from_context)
graph.add_node("no_answer_found", no_answer_found)
graph.add_node("is_sup", is_sup)
graph.add_node("revise_answer", revise_answer)
graph.add_node("is_use", is_use)
graph.add_node("rewrite_question", rewrite_question)

graph.add_edge(START, "decide_retrieval")
graph.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"retrieve": "retrieve", "generate_direct": "generate_direct"},
)
graph.add_edge("retrieve", "is_relevant")
graph.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found",
    },
)
graph.add_edge("generate_direct", END)
graph.add_edge("no_answer_found", END)
graph.add_edge("generate_from_context", "is_sup")
graph.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {"accept_answer": "is_use", "revise_answer": "revise_answer"},
)
graph.add_edge("revise_answer", "is_sup")
graph.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "end": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)
graph.add_edge("rewrite_question", "retrieve")

memory = MemorySaver()
chatbot = graph.compile(checkpointer=memory)
