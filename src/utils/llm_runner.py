from llama_cpp import Llama
# from langchain_community.llms import LlamaCpp


class Llm:
    """
    Wrapper class around llama.cpp for managing a local LLM instance.
    Provides an easy interface for generating chat-based responses.
    """
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, n_threads: int = 0):
        """
        Initialize the LLM model.

        Args:
            model_path (str): Path to the GGUF/GGML model file.
            n_ctx (int): Context window size (number of tokens). Default is 4096.
            n_gpu_layers (int): Number of layers to offload to GPU (-1 = all layers if supported).
            n_threads (int): Number of CPU threads to use (0 = auto-detect).
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,  
            chat_format="gemma"
        )
        # self.llm = LlamaCpp(
        #     model_path=model_path,
        #     temperature=0.75,
        #     max_tokens=2000,
        #     top_p=1,
        # )

    def generate_response(self, system_prompt :str , user_prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """
        Generate a chat response from the model.

        Args:
            system_prompt (str): Instruction/context given to the model (sets behavior).
            user_prompt (str): The user’s query or message.
            max_tokens (int): Maximum tokens to generate in the reply.
            temperature (float): Controls randomness (higher = more creative).

        Returns:
            str: The generated response from the model.
        """
        messages = [
                {
                    "role":"system",
                    "content":system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return out["choices"][0]["message"]["content"]

        # out = self.llm.invoke(messages)
        # return out
    

