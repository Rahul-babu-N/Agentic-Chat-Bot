from src.inference.run_chat_bot import ChatBot
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Global Setup
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_PATH = "models/Llama-3.2-3B-Instruct-IQ4_XS.gguf"
THREAD_ID = "2"

# HuggingFace Embeddings
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Persistent Vector Store
recall_vector_store = InMemoryVectorStore(hf)

# Persistent ChatBot
chat = ChatBot(model_path=MODEL_PATH)

# Config (can be expanded later)
config = {"configurable": {"thread_id": THREAD_ID}}

#compiled workflow
app = chat.compile(vector_store=recall_vector_store)
reply = chat.run(app=app, user_prompt="What is current time in INDIA", config=config)
print(reply)