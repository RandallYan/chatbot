from langchain.agents import tool

from chatbot_agent import ChatbotAgent

from tools.faq import get_faq_chain

# Define tools using @tool decorator
@tool("FAQ")
def faq(input_str: str) -> str:
    """Useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return get_faq_chain(input_str)

# @tool("Recommend Product")
# def recommend_product(input_str: str) -> str:
#     """Useful for when you need to search and recommend products and recommend it to the user"""
#     return product_chain()(input_str)

# @tool("Search Order", return_direct=True)
# def search_order(input_str: str) -> str:
#     """Useful for when you need to answer questions about customers orders"""
#     return orders.get_order(input_str)

if __name__ == "__main__":
    # Initialize the ChatbotAgent with the tools, LLM, and memory instance
    tools = [faq]
    agent = ChatbotAgent(tools=tools)

    # # 1. Can you deliver your goods to Shanghai? How many days would it take?

    # # Run the agent
    # while True:
    #     # if user input is "exit", exit the program
    #     question = input("User: ")
    #     if question == "exit":
    #         break
    #     # else, run the agent
    #     answer = agent.run(question)
    #     print("Bot: " + answer)

    agent.launch()
