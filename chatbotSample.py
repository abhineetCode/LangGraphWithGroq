import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display


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

def chatbot(state:State):
    return {"messages": llm.invoke(state["messages"])}
    #it is invoking the user query and returning the messages

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass


while True:
  user_input = input("User: ")
  if user_input.lower() in ['quit', 'q']:
    print("Chat Ended")
  events = graph.stream({"messages": ("user", user_input)}, stream_mode="values")
  for event in events:
    event["messages"][-1].pretty_print()