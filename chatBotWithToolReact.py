import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langchain_tavily import TavilySearch

os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
os.environ["LANGCHAIN_PROJECT"] = "myFirstlanggraph"
llm = ChatGroq(groq_api_key = "", model_name = "Gemma2-9b-It")

def multiplyTool(a:int,b:int) -> int:
    """Multiply a and b
    Args:
        a(int): first
        b(int): second
    Return:
        int: outpu int
    """
    return a * b

searchTool = TavilySearch(max_results = 2)
tools = [searchTool, multiplyTool]
llm_with_tool = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
def chatbot(state:State):
    return {"messages":llm_with_tool.invoke(state["messages"])}

graph_builder.add_node(chatbot, "chatbot")
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")

graph= graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


while True:
  user_input = input("User: ")
  if user_input.lower() in ['quit', 'q']:
    print("Chat Ended")
    break
  events = graph.stream({"messages": ("user", user_input)}, stream_mode="values")
  for event in events:
    event["messages"][-1].pretty_print()

