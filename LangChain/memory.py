import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print("AI:", conversation.predict(
    input="My name is Sangeeth. Reply in one short sentence."
))

print("AI:", conversation.predict(
    input="What is my name? Reply in one short sentence."
))




#------------------------------------simple-------------------------------------------
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# chat_history = []
# llm = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0
# )

# user_msg = "My name is Narsimha"
# chat_history.append(HumanMessage(content=user_msg))

# response = llm.invoke(chat_history)
# AI_message = chat_history.append(AIMessage(content=response.content))
# print(response.content)

# user_msg = "What is my Name ?"
# chat_history.append(HumanMessage(content=user_msg))

# response = llm.invoke(chat_history)
# AI_message = chat_history.append(AIMessage(content=response.content))

# print(response. Content)

