from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


class MyCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to llm events."""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n[Callback] LLM is starting...")
        print(f"[Callback] Prompt: {prompts}")

    def on_llm_new_token(self, token: str, **kwargs):
        print(f"[Callback] New token: {token}", end="", flush=True)

    
    def on_llm_end(self, response, **kwargs):
        print("\n[Callback] LLM finished.")
        print(f"[Callback] Response: \n{response.generations[0][0].text}")

    def on_llm_error(self, error, **kwargs):
        print(f"[Callback] Error: {error}")


llm = ChatGroq(groq_api_key = "", 
               model_name = "llama-3.3-70b-versatile", streaming=True)

def call_llm_with_callback(topic):
    """Call the language model with the given prompt and a standard output callback handler.
    
    Args:
        prompt (str): The input prompt for the language model.
        
    Returns:
        str: The response from the language model.
    """
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short poem about {topic}.")

    final_prompt = prompt.format(topic=topic)
    print("Generated Prompt:", final_prompt)
    chain = prompt | llm
    return chain.invoke(input={"topic": final_prompt}, config={"callbacks":[MyCallbackHandler()]})



while True:
    user_input = input("Topic: ")
    if user_input.lower() in ['quit', 'q']:
        print("Chat Ended")
        break
  
    response = call_llm_with_callback(user_input)
    print("\nPoem:", response)