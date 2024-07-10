# llm.py

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama LLM model with llama3
ollama_model = Ollama(model="llama3")
output_parser = StrOutputParser()

def chat(prompt):

    response = ollama_model.invoke(prompt)
    return output_parser.parse(response)