import uuid

import streamlit as st

from backend import chatbot

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# Styling
# ---------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:#0E1117;
}

.main-title{
    text-align:center;
    color:#4CAFEF;
    font-size:3rem;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#9AA0A6;
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🩺 MedAssist AI
</div>

<div class="subtitle">
Medical Knowledge Assistant using RAG + LangGraph
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session Helpers
# ---------------------------

def generate_thread():
    return str(uuid.uuid4())


def save_thread(thread_id):
    st.session_state["thread-history"].append(
        thread_id
    )
    return thread_id


def reset_chat():

    st.session_state["thread-id"] = save_thread(
        generate_thread()
    )

    st.session_state["chat-history"] = []


def get_config():

    return {
        "configurable": {
            "thread_id":
                st.session_state["thread-id"]
        }
    }


def get_custom_config(thread_id):

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def get_role(message: BaseMessage):

    if isinstance(message, HumanMessage):
        return "user"

    return "assistant"


# ---------------------------
# Session State
# ---------------------------

if "thread-history" not in st.session_state:
    st.session_state["thread-history"] = []

if "chat-history" not in st.session_state:
    st.session_state["chat-history"] = []

if "thread-id" not in st.session_state:

    st.session_state["thread-id"] = save_thread(
        generate_thread()
    )

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:

    st.header("Chats")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        reset_chat()

    st.divider()

    for thread_id in st.session_state[
        "thread-history"
    ][::-1]:

        state = chatbot.get_state(
            get_custom_config(thread_id)
        )

        title = "New Conversation"

        if (
            "messages" in state.values
            and len(state.values["messages"]) > 0
        ):

            title = " ".join(
                state.values["messages"][-1]
                .content.split()[:8]
            )

        if st.button(
            title,
            key=thread_id,
            use_container_width=True
        ):

            st.session_state["thread-id"] = thread_id
            st.session_state["chat-history"] = []

            state = chatbot.get_state(
                get_custom_config(thread_id)
            )

            if "messages" in state.values:

                for msg in state.values[
                    "messages"
                ]:

                    st.session_state[
                        "chat-history"
                    ].append({
                        "role": get_role(msg),
                        "content": msg.content
                    })

# ---------------------------
# Chat History
# ---------------------------

for msg in st.session_state["chat-history"]:

    avatar = (
        "👤"
        if msg["role"] == "user"
        else "🩺"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):
        st.write(msg["content"])

# ---------------------------
# Suggestions
# ---------------------------

if len(
    st.session_state["chat-history"]
) == 0:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button(
            "What is Diabetes?"
        )

    with col2:
        st.button(
            "Symptoms of Hypertension?"
        )

    with col3:
        st.button(
            "What causes Asthma?"
        )

# ---------------------------
# Chat Input
# ---------------------------

query = st.chat_input(
    "Ask a medical question..."
)

if query:

    st.session_state[
        "chat-history"
    ].append({
        "role": "user",
        "content": query
    })

    with st.chat_message(
        "user",
        avatar="👤"
    ):
        st.write(query)

    with st.spinner(
        "Searching medical knowledge..."
    ):

        response = chatbot.invoke(
            {
                "messages": [query]
            },
            config=get_config()
        )

    answer = (
        response["messages"][-1]
        .content
    )

    st.session_state[
        "chat-history"
    ].append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message(
        "assistant",
        avatar="🩺"
    ):
        st.write(answer)

# ---------------------------
# Disclaimer
# ---------------------------

st.divider()

st.warning(
    """
    This assistant provides educational medical
    information only and should not replace
    professional medical advice, diagnosis,
    or treatment.
    """
)
