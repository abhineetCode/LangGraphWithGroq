import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
os.environ["LANGCHAIN_PROJECT"] = "myFirstlanggraph"

llm = ChatGroq(groq_api_key = "", model_name = "Gemma2-9b-It")
#print(llm)

class State(TypedDict):
    #create a variable messages of type annotated that sas this is list and function call add_messages 
    #that add/append messages to the list on everyadd_messages teh state is updated
    messages:Annotated[list, add_messages]

graph_builder = StateGraph(State)
#print(graph_builder)

toolSearch = TavilySearch(max_results = 2)
def multiplyTool(a:int, b:int) -> int:
   """Multiply a and b

   Args:
        a(int) first int
        b(int) second int

    Returns
        int: output int   
   """
   return a * b

tools = [toolSearch, multiplyTool]
llm_with_tool = llm.bind_tools(tools)

def chatbot(state:State):
    return {"messages": llm_with_tool.invoke(state["messages"])}
    #it is invoking the user query and returning the messages

graph_builder.add_node("tool_calling_llm", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "tool_calling_llm")
graph_builder.add_conditional_edges("tool_calling_llm",  tools_condition)
graph_builder.add_edge("tools", END)

graph = graph_builder.compile()

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
