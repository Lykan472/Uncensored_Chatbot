import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Load variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# 1. Page Config
st.set_page_config(page_title="Dolphin-Mistral Chatbot", layout="wide")

# 2. Initialize the AI Model
# Using your requested model: cognitivecomputations/dolphin-mistral-24b-venice-edition:free
llm = ChatOpenAI(
    model="cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 3. Setup LangGraph Brain
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# We cache the graph so it doesn't rebuild on every click
@st.cache_resource
def get_chatbot():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    # MemorySaver keeps chat in RAM during the session
    return workflow.compile(checkpointer=MemorySaver())

chatbot = get_chatbot()

# 4. Sidebar - Chat Management
st.sidebar.title("💬 Chat Sessions")
if "threads" not in st.session_state:
    st.session_state.threads = {"Default Chat": "thread_1"}

new_chat_name = st.sidebar.text_input("New Chat Name")
if st.sidebar.button("➕ Create New Chat"):
    if new_chat_name:
        st.session_state.threads[new_chat_name] = f"thread_{len(st.session_state.threads)+1}"

selected_chat = st.sidebar.selectbox("Switch Chat", list(st.session_state.threads.keys()))
current_thread_id = st.session_state.threads[selected_chat]

# 5. Main Chat UI
st.title(f"🐬 {selected_chat}")

# Load history from the graph for this specific thread
config = {"configurable": {"thread_id": current_thread_id}}
state = chatbot.get_state(config)
chat_history = state.values.get("messages", []) if state.values else []

# Display history
for msg in chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Handle Input
if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
        st.write(prompt)
    
    # Run the AI
    input_data = {"messages": [HumanMessage(content=prompt)]}
    output = chatbot.invoke(input_data, config)
    
    # Display AI Response
    with st.chat_message("assistant"):
        ai_response = output["messages"][-1].content
        st.write(ai_response)