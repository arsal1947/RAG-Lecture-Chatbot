import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(page_title="Lecture Chatbot", page_icon="📚")
st.title("📚 Lecture Chatbot")
st.caption("Upload your lecture PDFs and ask questions about them.")

# ─── Session state setup ──────────────────────────────────────────────────────
# Session state persists data across reruns of the app
# Without it, variables reset every time the user interacts with the UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores all past messages

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None    # stores the RAG chain once built

# ─── Sidebar: PDF uploader ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Lectures")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True  # allows uploading multiple PDFs at once
    )

    # Button to trigger building the RAG pipeline
    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

# ─── Build RAG pipeline when button is clicked ────────────────────────────────
if process_btn and uploaded_files:
    with st.spinner("Processing your PDFs... this may take a minute ⏳"):

        # Step 1: Save uploaded files to a temp folder and load them
        all_documents = []
        for uploaded_file in uploaded_files:
            # Streamlit uploads are in memory, so we save them temporarily to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_documents.extend(docs)
            os.unlink(tmp_path)  # delete temp file after loading

        # Step 2: Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ""],
            chunk_size=200,
            chunk_overlap=20,
        )
        chunked_data = splitter.split_documents(all_documents)

        # Step 3: Create embeddings model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Step 4: Delete old ChromaDB and create fresh one
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        vectorstore = Chroma.from_documents(
            documents=chunked_data,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        # Step 5: Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Step 6: Create LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

        # Step 7: Create prompt
        prompt_template = """
You are a helpful study assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not use any outside knowledge.

Context: {context}

Question: {question}

Answer:
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Step 8: Build the LCEL chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        st.session_state.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Clear old chat history when new PDFs are uploaded
        st.session_state.chat_history = []

    st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) processed! {len(chunked_data)} chunks created.")

# ─── Display chat history ─────────────────────────────────────────────────────
# Loop through all past messages and display them
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):   # "user" or "assistant"
        st.markdown(message["content"])

# ─── Chat input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your lectures...")

if question:
    # Check if PDFs have been processed first
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload and process your PDFs first using the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Save user message to history
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Get answer from RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(question)
                st.markdown(answer)

        # Save assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
