
#-----------------------------------------------RunnableParallel------------------------------------
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm =  ChatGroq(model="openai/gpt-oss-20b",temperature=0)
parser = StrOutputParser()
summary_prompt = PromptTemplate.from_template(
    "Summary {Topic} in 3 simple lines"
)
example_prompt = PromptTemplate.from_template(
    "give 3 real world examples of {Topic}"
)
quiz_prompt = PromptTemplate.from_template(
    "Create 3 quiz questions about {Topic}"
)
summary_chain = summary_prompt | llm | parser
example_chain = example_prompt | llm | parser
quiz_chain = quiz_prompt | llm | parser

parller_chain = RunnableParallel(
    {
        "summary":summary_chain,
        "example":example_chain,
        "Quiz":quiz_chain
    }
)

response = parller_chain.invoke({ "Topic" : "AI"})
# print(response)

#-----------------------------------Runnable passthrough ------------------------------------------
summary_prompt = PromptTemplate.from_template(
    "Summary {Topic} in 3 simple lines"
)
llm =  ChatGroq(model="openai/gpt-oss-20b",temperature=0)
parser = StrOutputParser()

summary_prompt = PromptTemplate.from_template(
    "Summarize {Topic} in 3 simple lines."
)

summary_chain = summary_prompt | llm | parser

chain = RunnablePassthrough.assign(
    Summary=summary_chain
)

result = chain.invoke({"Topic": "LangChain"})

print("Original Topic:", result["Topic"])
print("Original Summary:", result["Summary"])



#-------------------------------------Runnable Sequencial ----------------------------------------------#

prompt = PromptTemplate.from_template( """
Analyze this restaurant review.

Review: {review}

Return only JSON with these keys:
{{
  "sentiment": "Positive / Negative / Neutral",
  "reason": "short reason",
  "rating": "rating out of 5"
}}

""")

model = ChatGroq(model="openai/gpt-oss-20b",temperature=0)

parser = JsonOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "review": "Food was tasty but delivery was very late."
})

# print(result)
# print(result["sentiment"])
# print(result["reason"])
# print(result["rating"])