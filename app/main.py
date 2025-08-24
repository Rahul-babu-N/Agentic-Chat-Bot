from src.inference.run_chat_bot import ChatBot
import gradio as gr
from typing import List
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
    encode_kwargs={"normalize_embeddings": True}
)

# Persistent Vector Store
recall_vector_store = InMemoryVectorStore(hf)

# Persistent ChatBot
chat = ChatBot(model_path=MODEL_PATH)

# Config (can be expanded later)
config = {"configurable": {"thread_id": THREAD_ID}}

#compiled workflow
app = chat.compile(vector_store=recall_vector_store)

def respond(history: List[List[str]], user_msg: str):
    """Handles chatbot response and updates history."""
    reply = chat.run(app=app, user_prompt=user_msg, config=config)
    history.append([user_msg, reply])
    return history, ""

with gr.Blocks(title="Local LLM Chatbot") as demo:
    gr.Markdown("# Local LLM Chatbot \nRuns fully local.")
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message...")
        send = gr.Button("Send")
    clear = gr.Button("Clear")

    send.click(respond, [chatbot, msg], [chatbot, msg])
    msg.submit(respond, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)


