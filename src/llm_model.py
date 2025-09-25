from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel, BaseLLM
from typing import Union

def get_ollama_llm(model_name: str, base_url: str = "http://localhost:11434") -> Union[BaseLLM, BaseChatModel]:
    print(f"Initializing Ollama LLM with model: '{model_name}' at '{base_url}'")
    try:
        llm = ChatOllama(model=model_name, base_url=base_url)
        print("ChatOllama model initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing ChatOllama (falling back to Ollama LLM if possible): {e}")
        try:
            llm = Ollama(model=model_name, base_url=base_url)
            print("Ollama LLM model initialized successfully (using BaseLLM).")
            return llm
        except Exception as fallback_e:
            print(f"Error initializing Ollama LLM (even fallback failed): {fallback_e}")
            print("Please ensure Ollama is running and the specified LLM model is pulled.")
            raise # Re-raise the exception if no LLM can be initialized

