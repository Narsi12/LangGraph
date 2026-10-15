from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-20b",temperature=0)
response = model.invoke("Tell me about your capabilities")
print(response. Content)