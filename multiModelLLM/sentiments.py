import argparse
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key = "", 
               model_name = "llama-3.3-70b-versatile")

def create_prompt(text: str) -> str:
    """Create input prompt for the language model.
    
    Args:
        text: text to classify
        
    Returns:
        str: prompt for text classification.
    """
    instructions= "Is the underlying sentiments positive, negative?"
    formatting = '"Positive or Negative"'
    return f"Text:{text}\n{instructions}\nAnswer:{formatting}:"

def call_llm(prompt):
    """Call the language model with the given prompt.
    Args:
        prompt (str): The input prompt for the language model.
    Returns:
        str: The response from the language model."""
        
    messages = [{"role": "user", "content": prompt}]
    return llm.invoke(messages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis with Groq LLM")
    parser.add_argument("text", type=str,  help="Text to analyze sentiment for")
    args = parser.parse_args()
    
    prompt = create_prompt(args.text)
    response = call_llm(prompt)
    print("Model Response:", response.content)
        
    