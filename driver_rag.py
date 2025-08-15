import hashlib
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore



# ===== LOAD KEYS =====
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Set as environment variables so Pinecone + Gemini can see them
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ===== CONFIG =====
INDEX_NAME = "driver-index"
NAMESPACE = "driver-namespace"

# ===== INIT PINECONE =====
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")

# Connect to index
index = pc.Index(INDEX_NAME)

# ===== DELETE OLD DATA SAFELY =====
# try:
#     index.delete(delete_all=True, namespace=NAMESPACE)
#     print("✅ Old data deleted from Pinecone index.")
# except Exception as e:
#     if "Namespace not found" in str(e):
#         print(f"ℹ Namespace '{NAMESPACE}' does not exist yet — skipping delete.")
#     else:
#         raise e

# ===== EMBEDDINGS + LLM =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, google_api_key=GOOGLE_API_KEY)

# ===== DOCUMENTS =====
text = """
I visited Karachi first, then Lahore, and finally Islamabad.Karachi is a coastal city in Pakistan, Lahore is famous for its culture, and Islamabad is the capital city.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

##############################################################################################3
def generate_id_from_text(text, prefix="driver-doc"):
    # Hash banaye text ka
    hash_val = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]  # first 10 chars
    return f"{prefix}-{hash_val}"


doc_ids = [generate_id_from_text(doc.page_content) for doc in docs]

# ===== INSERT INTO PINECONE =====
vectorstore = PineconeVectorStore.from_documents(
    docs, embeddings, index_name=INDEX_NAME, namespace=NAMESPACE, ids=doc_ids
)
print("✅ New data inserted into Pinecone.")

# ===== SIMPLE SEARCH (NO HYBRID) =====
query = "madina"
retrieved_docs = vectorstore.similarity_search(query, k=3, namespace=NAMESPACE)

print("\n🔍 Retrieved Documents:\n")
for doc in retrieved_docs:
    print(doc.page_content)

# ===== GENERATE ANSWER =====

context = "".join([doc.page_content for doc in retrieved_docs])
print("context is: ", context)
prompt = f"""Answer the question strictly using only the context provided below. write answer according to the question don't just copy paste the context.
If the answer is not present in the context, reply with "Sorry sir Please ask relevant questions" and do not make up any information.

Context:
{context}

Question: {query}

Answer:"""

response = llm.predict(prompt)

print(f"\n💡 Answer: {response}")
