import streamlit as st
import time
from agentic_rag_graph import app

st.set_page_config(page_title="Enterprise Agentic RAG System", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Stateful RAG Dashboard")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a financial or general question..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.status("🤖 Orchestrating Agent Reasoning Graph...", expanded=True) as status_box:
            inputs = {"query": user_query, "loop_count": 0}
            final_response = ""
            
            for event in app.stream(inputs, stream_mode="updates"):
                for node_name, state_update in event.items():
                    if node_name == "router":
                        st.markdown(f"**🧭 Router Agent:** Directing flow to `[{state_update.get('route_to', 'unknown').upper()}]`")
                    elif node_name == "vectorstore":
                        st.markdown("**📚 Vector Store Retriever:** Fetching matches...")
                    elif node_name == "doc_grader":
                        st.markdown(f"**⚖️ Compliance Grader:** Context checked.")
                    elif node_name == "query_rewriter":
                        st.markdown(f"**🔄 Query Refinement Agent:** Rewriting search metrics.")
                    elif node_name == "generator":
                        final_response = state_update.get("generation", "")
            status_box.update(label="✅ Loop Complete", state="complete")
        response_placeholder.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
