import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- SECRETS MANAGEMENT ---
api_key = st.secrets["OPENROUTER_API_KEY"]

# 1. Page Config
st.set_page_config(page_title="Ultra-Resilient Chat", layout="wide")

# 2. Model Configuration
# Main model + 3 Fallbacks (Ranked by speed/stability)
MODELS = [
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free", # Main
    "google/gemini-2.0-flash-exp:free",                             # Fallback 1 (High Stability)
    "mistralai/mistral-7b-instruct:free",                           # Fallback 2 (Fast)
    "nousresearch/hermes-3-llama-3.1-405b:free"                             # Fallback 3 (Legacy)
]

def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=1, # We handle retries manually via cascading
        timeout=15
    )

# 3. Setup LangGraph Brain with Cascading Logic
def call_model(state: MessagesState):
    system_msg = SystemMessage(content="You are a helpful AI assistant.")
    messages = [system_msg] + state["messages"]
    
    # Cascade through models until one works
    for i, model_name in enumerate(MODELS):
        try:
            llm = get_llm(model_name)
            response = llm.invoke(messages)
            
            # Add a small badge if it used a fallback
            if i > 0:
                response.content = f"*(Failsafe Level {i} Active)*\n\n" + response.content
            
            return {"messages": [response]}
        except Exception as e:
            print(f"Model {model_name} failed. Error: {e}")
            continue # Move to the next model in the list
            
    # If the loop finishes without returning, everything failed
    raise Exception("All 4 models are currently unavailable.")

@st.cache_resource
def get_chatbot():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    return workflow.compile(checkpointer=MemorySaver())

chatbot = get_chatbot()

# 4. Sidebar - Chat Management
st.sidebar.title("🐬 Session Manager")
if "threads" not in st.session_state:
    st.session_state.threads = {"Primary Chat": "thread_1"}

selected_chat = st.sidebar.selectbox("Select Chat", list(st.session_state.threads.keys()))
current_thread_id = st.session_state.threads[selected_chat]

if st.sidebar.button("🗑️ Reset Current Conversation"):
    st.session_state.threads[selected_chat] = f"thread_{current_thread_id}_new"
    st.rerun()

# 5. Chat UI Logic
st.title(f"Conversing in: {selected_chat}")

config = {"configurable": {"thread_id": current_thread_id}}
state = chatbot.get_state(config)
chat_history = state.values.get("messages", []) if state.values else []

# Display Messages
for msg in chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Input Box
if prompt := st.chat_input("Ask anything..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # The 'invoke' triggers the call_model function which handles the 4 models
            output = chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config)
            ai_response = output["messages"][-1].content
            message_placeholder.write(ai_response)
        except Exception:
            # The final user-friendly message
            message_placeholder.error("🛑 **System Overloaded.** All four free model providers are currently down. Please wait 1 minute and try again.")
