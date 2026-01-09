import streamlit as st
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- SECRETS MANAGEMENT ---
api_key = st.secrets["OPENROUTER_API_KEY"]

# 1. Page Config
st.set_page_config(page_title="Advanced Chatbot", layout="wide")

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

# 4. DIALOGS (Pop-ups for Editing)
@st.dialog("Rename Chat")
def rename_chat_dialog(old_name):
    new_name = st.text_input("New chat name", value=old_name)
    if st.button("Save Name"):
        if new_name and new_name != old_name:
            st.session_state.threads[new_name] = st.session_state.threads.pop(old_name)
            if st.session_state.current_thread_name == old_name:
                st.session_state.current_thread_name = new_name
            st.rerun()

@st.dialog("Edit Message")
def edit_message_dialog(index, old_content, thread_id):
    new_content = st.text_area("Edit your message", value=old_content, height=200)
    if st.button("Update Message"):
        # Access the internal state of LangGraph to update history
        current_state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        messages = list(current_state.values["messages"])
        
        # Update the message content
        if isinstance(messages[index], HumanMessage):
            messages[index] = HumanMessage(content=new_content)
        else:
            messages[index] = AIMessage(content=new_content)
            
        # Push the updated history back to the database
        chatbot.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": messages},
            as_node="agent"
        )
        st.rerun()

# 5. SIDEBAR - CHAT MANAGEMENT
st.sidebar.title("💬 Chat History")

if "threads" not in st.session_state:
    st.session_state.threads = {"New Conversation": str(uuid.uuid4())}
if "current_thread_name" not in st.session_state:
    st.session_state.current_thread_name = "New Conversation"

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.threads[f"Chat {len(st.session_state.threads)+1}"] = new_id
    st.session_state.current_thread_name = list(st.session_state.threads.keys())[-1]

st.sidebar.button("➕ New Chat", on_click=create_new_chat, use_container_width=True)
st.sidebar.markdown("---")

for chat_name in list(st.session_state.threads.keys()):
    col1, col2, col3 = st.sidebar.columns([0.6, 0.2, 0.2])
    if col1.button(chat_name, key=f"sel_{chat_name}", use_container_width=True):
        st.session_state.current_thread_name = chat_name
        st.rerun()
    if col2.button("✏️", key=f"rename_{chat_name}"):
        rename_chat_dialog(chat_name)
    if col3.button("🗑️", key=f"del_{chat_name}"):
        del st.session_state.threads[chat_name]
        if not st.session_state.threads: create_new_chat()
        st.session_state.current_thread_name = list(st.session_state.threads.keys())[0]
        st.rerun()

# 6. MAIN CHAT UI
active_thread_name = st.session_state.current_thread_name
active_thread_id = st.session_state.threads[active_thread_name]
st.title(f"🐬 {active_thread_name}")

config = {"configurable": {"thread_id": active_thread_id}}
state = chatbot.get_state(config)
chat_history = state.values.get("messages", []) if state.values else []

# Display Messages with Inline Edit Buttons
for i, msg in enumerate(chat_history):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        col_text, col_edit = st.columns([0.9, 0.1])
        col_text.markdown(msg.content)
        if col_edit.button("📝", key=f"edit_msg_{i}"):
            edit_message_dialog(i, msg.content, active_thread_id)

if prompt := st.chat_input("Ask anything..."):
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
            message_placeholder.markdown(output["messages"][-1].content)
        except Exception:
            message_placeholder.error("🛑 System Overloaded. Try again later.")
