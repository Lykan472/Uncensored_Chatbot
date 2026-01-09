import streamlit as st
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- SECRETS MANAGEMENT ---
api_key = st.secrets["OPENROUTER_API_KEY"]

# 1. Page Config
st.set_page_config(page_title="Ultra-Resilient Chat", layout="wide")

# 2. Model Configuration
MODELS = [
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free"
]

def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=1,
        timeout=15
    )

# 3. Setup LangGraph Brain
def call_model(state: MessagesState):
    system_msg = SystemMessage(content="You are a helpful AI assistant.")
    messages = [system_msg] + state["messages"]
    
    for i, model_name in enumerate(MODELS):
        try:
            llm = get_llm(model_name)
            response = llm.invoke(messages)
            if i > 0:
                response.content = f"*(Failsafe Level {i} Active)*\n\n" + response.content
            return {"messages": [response]}
        except Exception:
            continue
    raise Exception("All models unavailable.")

@st.cache_resource
def get_chatbot():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    return workflow.compile(checkpointer=MemorySaver())

chatbot = get_chatbot()

# 4. SIDEBAR - CHAT MANAGEMENT
st.sidebar.title("💬 Chat History")

# Initialize threads in session state if not present
if "threads" not in st.session_state:
    st.session_state.threads = {"New Conversation": str(uuid.uuid4())}
if "current_thread_name" not in st.session_state:
    st.session_state.current_thread_name = "New Conversation"

# Function to create a new chat
def create_new_chat():
    new_id = str(uuid.uuid4())
    chat_num = len(st.session_state.threads) + 1
    new_name = f"Chat {chat_num}"
    st.session_state.threads[new_name] = new_id
    st.session_state.current_thread_name = new_name

st.sidebar.button("➕ New Chat", on_click=create_new_chat, use_container_width=True)

# Sidebar selection list (simulating ChatGPT sidebar)
st.sidebar.markdown("---")
for chat_name in list(st.session_state.threads.keys()):
    # Highlight the currently active chat
    if st.sidebar.button(chat_name, key=chat_name, use_container_width=True):
        st.session_state.current_thread_name = chat_name
        st.rerun()

# 5. MAIN CHAT UI
active_thread_name = st.session_state.current_thread_name
active_thread_id = st.session_state.threads[active_thread_name]

st.title(f"🐬 {active_thread_name}")

config = {"configurable": {"thread_id": active_thread_id}}
state = chatbot.get_state(config)
chat_history = state.values.get("messages", []) if state.values else []

# Display Messages
for msg in chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input Box
if prompt := st.chat_input("Ask anything..."):
    # Append the first message to the name if it's still 'New Conversation'
    if active_thread_name == "New Conversation":
        new_name = prompt[:20] + "..." if len(prompt) > 20 else prompt
        st.session_state.threads[new_name] = st.session_state.threads.pop("New Conversation")
        st.session_state.current_thread_name = new_name
        st.rerun()

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            output = chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config)
            ai_response = output["messages"][-1].content
            message_placeholder.markdown(ai_response)
        except Exception:
            message_placeholder.error("🛑 **System Overloaded.** Please try again later.")
