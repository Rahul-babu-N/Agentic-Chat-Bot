from src.inference.run_chat_bot import ChatBot
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from config.config import CONFIG

# Global Setup
EMBEDDING_MODEL = CONFIG["CHATBOT_DETAILS"]["EMBEDDING_MODEL"]
MODEL_PATH = CONFIG["CHATBOT_DETAILS"]["MODEL_PATH"]
THREAD_ID = CONFIG["CHATBOT_DETAILS"]["THREAD_ID"]

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
reply = chat.run(app=app, user_prompt="What is the capital of france", config=config)
print(reply)