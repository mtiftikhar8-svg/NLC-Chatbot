import streamlit as st
import os
import sys
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# -------------------- LOAD ENV --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------- STREAMLIT PAGE CONFIG --------------------
st.set_page_config(page_title="💎 Gorgeous RAG (Gemini + Pinecone)", page_icon="💎")
st.title("💎 NLC ChatBot")
st.write("Upload PDFs and query to our CHatbot.")

# -------------------- WINDOWS ASYNC FIX --------------------
def ensure_event_loop():
    """Fix for 'no running event loop' error in Streamlit on Windows."""
    if sys.platform.startswith("win"):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

# -------------------- INITIALIZE PINECONE --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "gorgeous-rag"

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# -------------------- HELPERS --------------------
def insert_text_to_pinecone(text):
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    vector_store = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
    vector_store.add_documents(docs)

def search_and_answer(query):
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
   
    prompt = f"""
                You are a polite Urdu/English bilingual customer-support chatbot for a RAG system.
                Follow these rules STRICTLY:

                1) GREETING-ONLY RESPONSES (no context lookup):
                - If user says "Hi" → reply: "Hi Sir, how may I help you?"
                - If user says "Hello" → reply: "Hello Sir, how may I help you?"
                - If user says "Assalam o Alaikum", "AOA", or "aoa" → reply: "Wa Alaikum Assalam Sir, how may I help you?"
                - If user asks about your well-being ("How are you?", "Kya haal hai?", etc.) → reply: "I am good Sir. Hope you are doing well as well. How may I help you?"

                2) PRAISE / COMPLIMENT RESPONSES (no context lookup):
                - If the user compliments you (e.g., "good bot", "nice work", "you are great", "shukriya", "thanks", etc.):
                    - Reply politely with gratitude and warmth.
                    - Example: "Thank you very much Sir, I really appreciate it. If you want any further assistance, let me know."

                3) QUESTION ABOUT CAPABILITIES:
                - If the user asks "what type of questions can I ask?", "what can you do?", "how can you help me?", or similar:
                    - Scan the provided context to understand the main topic or domain (e.g., company, project, document type).
                    - Reply with a polite, short introduction about what you can provide information on, based on that context.
                    - Example: "Sir, I can assist you with information about NLC’s services, projects, and operations as mentioned in the provided data."

                4) NON-GREETING QUERIES:
                - Always check if there is any relevant information in the provided context — even if the question is vague (e.g., "explain this file", "which sensors are used in this project").
                - If relevant details exist in context:
                    - Start reply with "Sir" and clearly explain the answer in polite, concise language.
                    - If needed, summarize and highlight key points in bullet form.
                - If no relevant details exist in context:
                    - Reply: "Sorry Sir, I can’t help you with this."

                5) Do NOT invent, guess, or add information not present in the provided context.

                --------------------
                CONTEXT:
                {context}
                --------------------

                USER QUESTION:
                {query}

                Now produce your reply.
            """




    response = llm.invoke(prompt)
    return response.content

# -------------------- UPLOAD SECTION --------------------
uploaded_files = st.file_uploader("📄 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
# text_input = st.text_area("✏️ Or enter text to store in Pinecone:")

if st.button("Insert Data"):
    with st.spinner("📥 Inserting PDFs into Pinecone... Please wait."):
        if uploaded_files:
            for file in uploaded_files:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                insert_text_to_pinecone(text)
            st.success("✅ PDFs inserted into Pinecone.")
        # elif text_input.strip():
        #     insert_text_to_pinecone(text_input)
        #     st.success("✅ Text inserted into Pinecone.")
        else:
            st.error("Please upload a PDF or enter some text.")

# -------------------- QUERY SECTION --------------------
query_input = st.text_input("🔍 Ask a question:")

if query_input.strip():  # Runs automatically when user presses Enter
    with st.spinner("🔍 Chatbot is typing..."):
        answer = search_and_answer(query_input)
    st.info(answer)