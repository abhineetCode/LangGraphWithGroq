from langchain.callbacks import StdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key = "", 
               model_name = "llama-3.3-70b-versatile", streaming=True)

def call_llm_with_callback(prompt):
    """Call the language model with the given prompt and a standard output callback handler.
    
    Args:
        prompt (str): The input prompt for the language model.
        
    Returns:
        str: The response from the language model.
    """

    """ messages = [
            {"role": "system", "content": "You are a helpful assistant that provides detailed responses."},
            {"role": "user", "content": prompt},] """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides detailed responses."),
        ("user", "{prompt}")
    ])

    chain = prompt_template | llm

    callback_handler = StdOutCallbackHandler()
    return chain.invoke(input={"prompt": prompt}, config={"callbacks":[callback_handler]})


while True:
  user_input = input("User: ")
  if user_input.lower() in ['quit', 'q']:
    print("Chat Ended")
    break
  result = call_llm_with_callback(user_input)
  print(result)




