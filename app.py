import streamlit as st
import tempfile
import os
from rag_local import load_and_split, create_vectorstore, create_graph
from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages

st.title("📄 Document Assistant")

uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)

messages: Annotated[Sequence[BaseMessage], add_messages] = []
output = {"messages": messages}

if "messages" not in st.session_state:
    st.session_state.messages = []


if uploaded_files:
    with st.spinner("Reading and indexing..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False,prefix=f"{uploaded_file.name.replace('.pdf','')}___", suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            
            chunks = load_and_split(tmp_file_path)
            all_chunks += chunks
        vectorstore = create_vectorstore(all_chunks)
        graph = create_graph(vectorstore)
    
    st.success("Documents ready! Ask a question.")

    query = st.text_input("Ask something about the document:")



    if query:
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": "abc345"}}   
            messages = st.session_state.messages
            messages.append(HumanMessage(content=query))
            output = graph.invoke({"messages": messages}, config)
            st.session_state.messages = output["messages"]

        st.markdown("### 💬 Chat History")

        for msg in st.session_state.messages:
            if msg.type == "human":
                st.markdown(f"**🧑 You:** {msg.content}")
            elif msg.type == "ai":
                st.markdown(f"**🤖 Assistant:** {msg.content}")

        


        with st.expander("Sources"):
            for idx, doc in enumerate(output["context"], start=1):
                # Build the full path (you might need to adjust depending where your PDFs are served from)
                source_name = os.path.basename(doc.metadata["source"]).split("___")[0]
                
                # Example: if you serve PDFs from a local server or a static folder, build a link like:
                # pdf_url = doc.metadata["source"]  # <-- you MUST have a way to serve this file and then use [link text](URL) inside the markdown
                
                st.markdown(f"**Source {idx}: {source_name}.pdf**")

                st.markdown(f"Page `{doc.metadata['page']}`, Start Position `{doc.metadata['start_index']}`")
                st.write(doc.page_content)
                st.markdown("---")  # Separator


   
                  


    # Optional: clean up temp file
    os.remove(tmp_file_path)
