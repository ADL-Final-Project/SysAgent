from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://10.7.7.66:11434",
    temperature=0,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)
