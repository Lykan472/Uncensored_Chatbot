import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- SECRETS MANAGEMENT ---
api_key = st.secrets["OPENROUTER_API_KEY"]

# 1. Page Config & Custom Styling
st.set_page_config(page_title="Dolphin-Mistral Chatbot", layout="wide")

# 2. Define LLM Factory with Failsafe Logic
def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
        timeout=20
    )

# Primary and Fallback Models
PRIMARY_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free" # Highly stable free model

# 3. Setup LangGraph Brain
def call_model(state: MessagesState):
    system_msg = SystemMessage(content="You are Dolphin-Mistral, a helpful and uncensored AI assistant.")
    messages = [system_msg] + state["messages"]
    
    try:
        # Attempt Primary Model
        llm = get_llm(PRIMARY_MODEL)
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # Silently log error to console and try fallback
        print(f"Primary model failed: {e}")
        try:
            llm_fallback = get_llm(FALLBACK_MODEL)
            response = llm_fallback.invoke(messages)
            # Add a small note to the response so you know it switched
            response.content = "*(Switched to fallback)* " + response.content
            return {"messages": [response]}
        except Exception as final_error:
            # If both fail, raise an exception to be caught by the UI
            raise final_error

@st.cache_resource
def get_chatbot():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    return workflow.compile(checkpointer=MemorySaver())

chatbot = get_chatbot()

# 4. Sidebar - Session Management
st.sidebar.title("🐬 Dolphin Sessions")
if "threads" not in st.session_state:
    st.session_state.threads = {"Main Chat": "thread_1"}

selected_chat = st.sidebar.selectbox("Active Chat", list(st.session_state.threads.keys()))
current_thread_id = st.session_state.threads[selected_chat]

if st.sidebar.button("🗑️ Clear This Chat"):
    # Re-initialize the thread by changing its ID
    st.session_state.threads[selected_chat] = f"thread_{current_thread_id}_reset"
    st.rerun()

# 5. Main Chat UI
st.title(f"Chat: {selected_chat}")

config = {"configurable": {"thread_id": current_thread_id}}
state = chatbot.get_state(config)
chat_history = state.values.get("messages", []) if state.values else []

for msg in chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

if prompt := st.chat_input("Message Dolphin-Mistral..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            input_data = {"messages": [HumanMessage(content=prompt)]}
            output = chatbot.invoke(input_data, config)
            ai_response = output["messages"][-1].content
            message_placeholder.write(ai_response)
        except Exception as e:
            # CLEAN FAILSAFE MESSAGE: No red code logs
            message_placeholder.error("🚨 **Model not available.** The free servers are currently overloaded. Please try again in 30 seconds or switch sessions.")
