from src.inference.run_chat_bot import ChatBot
chat = ChatBot(model_path="models/gemma-3-270m-it-F16.gguf")
config = {"configurable" : { "thread_id" : '2' } }
reply = chat.run(user_prompt="what is my name",config=config)
print(reply)