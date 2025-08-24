
from src.utils.chat_bot_builder import ChatBotCompiler

class ChatBot:
    def __init__(self,model_path):
        """
        Initialize the chatbot with model path
        """
        self.model_path = model_path

    def compile(self, vector_store):
        """
        Compile and build the graph
        """
        chat_bot_compiler = ChatBotCompiler(model_path=self.model_path)
        app = chat_bot_compiler.build_graph(vector_store = vector_store)
        return app

    def run(self,app, user_prompt, config):
        """
        Executes a full chatbot workflow for a given user prompt.
        
        Args:
            app: Compiled graph.
            user_prompt (str): The input query from the user.
            config (RunnableConfig): Configuration object that manages workflow execution state.
             
        Returns:
            str: The chatbot’s final response text.
        """

        output = app.invoke({"user_prompt": user_prompt},config)
        return output["final_result"]

