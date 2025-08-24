from langgraph.graph import START, StateGraph, END
from langchain_community.tools import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
import os
from typing import List
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
import uuid
from src.utils.llm_runner import Llm
os.environ["TAVILY_API_KEY"] = "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"


class ChatBotCompiler:
    def __init__(self,model_path):
        # Initialize LLM wrapper
        self.llm = Llm(model_path=model_path)
        # Define the state schema for StateGraph
        class GraphState(TypedDict):
            user_prompt : str
            final_result: str
            is_web_search_required: bool
            web_search_query: str
            web_search_result: str
            recall_memories : List[str]
        # Build workflow graph with defined schema
        self.workflow = StateGraph(GraphState)
        # Web search tool (Tavily API wrapper)
        self.web_search_tool = TavilySearchResults(max_results=1,include_answer=True)


    def build_graph(self,vector_store):

        def load_memory(state):
            """
            Retrieve relevant conversational memory from vector_store.
            Uses similarity search to recall semantically related past context.
            """
            user_prompt = state["user_prompt"]
            query = f"Retrieve previous conversations or facts related to: {user_prompt}. \
            Focus on semantically similar topics, names, or preferences mentioned earlier."
            documents = vector_store.similarity_search(query, k=1)
            recall_memories =  [document.page_content for document in documents]
            return {
                "recall_memories": recall_memories
            }
        
        def decide(state):
            """
            Decide if a web search is needed based on user query.
            Uses the LLM with a strict system prompt to return 'yes' or 'no'.
            """

            user_prompt = state["user_prompt"]
            system_prompt = """
                You are a decision-making assistant.
                Decide if the user’s query requires a real-time web search.
                Use web search ONLY if:
                - The question is about recent events, factual data (dates, numbers, stats), or information not stored in memory.
                - The question requires external references (e.g., news, weather, live updates).

                If the query can be answered with reasoning, memory, or general knowledge, return "no".

                Answer STRICTLY with either "yes" or "no".
           
            """
            response = self.llm.generate_response(system_prompt=system_prompt,user_prompt=user_prompt)

            print(response,"\n\n")
            if response.lower() == "yes":
                
                return {"is_web_search_required":True}
            else:
               
                return {"is_web_search_required":False}
            
        def decide_route(state):
             # Conditional routing after "decide"
            is_web_search_required = state["is_web_search_required"]
            if is_web_search_required:
                
                return "web_search"
            else:
               
                return "model"
            
        def generate_web_search_query(state):

            """
            Convert user prompt into a concise search-friendly query.
            """

            user_prompt = state["user_prompt"]
            system_prompt = """
                You are a query rewriter for web search.
                Given the user’s request, generate a concise, search-friendly query
                that will return the most relevant results.

                Guidelines:
                - Use simple keywords (no extra words).
                - Avoid pronouns like "I" or "my".
                - If the user asks about a person/place/event, include full names and context.
                - Do not generate multiple sentences. Only output ONE query string.

            """
            response = self.llm.generate_response(system_prompt=system_prompt,user_prompt=user_prompt)
            return {"web_search_query": response}
        
        def web_search_tool(state):
            """
            Run Tavily search with the generated query.
            """
            web_search_query = state["web_search_query"]
            web_results = self.web_search_tool.invoke({'query': web_search_query})
            if web_results:
                return {"web_search_result": web_results[0].content}
            else:
                return {"web_search_result": "No results found."}
        
        def generate_final_response(state):
            """
            Generate the final chatbot response using:
            - Memory (recall_memories)
            - Web search results (if available)
            """

            system_prompt = """
                You are a knowledgeable, polite AI assistant. Follow these rules:
                - Provide accurate, concise, and well-structured answers.
                - If technical, explain step by step with examples.
                - If ambiguous, ask clarifying questions instead of guessing.
                - If uncertain or no information available, say: "I don’t know."
                - For code: return clean, formatted snippets.
                - For sensitive/harmful queries: politely refuse.

                When web search results are available, summarize them and integrate with reasoning
                instead of copying verbatim.
                When previous conversations are available, use them in geneerating the response if they are relevant.
                    """
            if not state["is_web_search_required"]:
                user_prompt = state["user_prompt"] 
                recall_mem = state["recall_memories"]
                final_user_prompt = (
                    f"user prompt : {user_prompt} "
                    f"and relevant past conversational memory in the form of "
                    f"{{user prompt : past chatbot response}} : {recall_mem}"
                )
                print(final_user_prompt,"\n\n")
                response = self.llm.generate_response(system_prompt=system_prompt,user_prompt=final_user_prompt)
            else:
                user_prompt = state["user_prompt"] 
                web_search_result = state["web_search_result"]
                recall_mem = state["recall_memories"]
                final_user_prompt = (
                f"user prompt : {user_prompt}, "
                f"relevant conversational memory in the form of "
                f"{{user prompt : past chatbot response}} : {recall_mem} "
                f"and web search results : {web_search_result}"
                )
                response = self.llm.generate_response(system_prompt=system_prompt,user_prompt=final_user_prompt)

            return {"final_result":response}
        
        def save_memory(state):
            """
            Save user prompt and chatbot response into vector_store for long-term recall.
            """
            
            user_prompt = state["user_prompt"]
            final_result = state["final_result"]
            doc = {
                "user prompt": user_prompt,  
                "chatbot response": final_result
                }
            _doc = Document(page_content=str(doc), id=str(uuid.uuid4()), metadata={"description": "Captured memory of past interactions"})
            vector_store.add_documents([_doc])
        
        self.workflow.add_node('load_memory',load_memory)
        self.workflow.add_node("decide",decide)
        self.workflow.add_node("generate_web_search_query",generate_web_search_query)
        self.workflow.add_node("web_search_tool",web_search_tool)
        self.workflow.add_node("generate_final_response",generate_final_response)
        self.workflow.add_node('save_memory',save_memory)
        self.workflow.set_entry_point('load_memory')
        self.workflow.add_edge("load_memory","decide")
        self.workflow.add_conditional_edges("decide",decide_route,{"web_search":"generate_web_search_query","model":"generate_final_response"})
        self.workflow.add_edge("generate_web_search_query","web_search_tool")
        self.workflow.add_edge("web_search_tool","generate_final_response")
        self.workflow.add_edge("generate_final_response","save_memory")

        memory = MemorySaver()
        app = self.workflow.compile(checkpointer=memory)

        return app
    
    # def save_memory(self,app, config: RunnableConfig,vector_store):
    #     """
    #     Save user prompt and chatbot response into vector_store for long-term recall.
    #     """
    #     mem = app.checkpointer.get(config).get("channel_values",{}) 
    #     user_prompt = mem.get("user_prompt")
    #     final_result = mem.get("final_result")
    #     doc = {
    #         "user prompt": user_prompt,  
    #         "chatbot response": final_result
    #         }
    #     _doc = Document(page_content=str(doc), id=str(uuid.uuid4()), metadata={"description": "Captured memory of past interactions"})
    #     vector_store.add_documents([_doc])


