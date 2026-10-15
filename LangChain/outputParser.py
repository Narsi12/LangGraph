from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

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

print(result)
print(result["sentiment"])
print(result["reason"])
print(result["rating"])