from langchain.agents import tool

from chatbot import Chatbot

from tools.faq import get_faq_chain

# Define tools using @tool decorator
@tool("FAQ")
def faq(input_str: str) -> str:
    """Useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return get_faq_chain(input_str)

if __name__ == "__main__":
    # Initialize the ChatbotAgent with the tools, LLM, and memory instance
    tools = [faq]

    agent = Chatbot(tools=tools)
    agent.launch()
