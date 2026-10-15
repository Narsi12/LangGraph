############################# Single-Turn Prompt Template #############################

from langchain_core.prompts import PromptTemplate

prompt1 = PromptTemplate.from_template(
    """
Explain {topic}
in simple Telugu.
"""
)

formatted_prompt1 = prompt1.format(
    topic="Agentic AI"
)

print(formatted_prompt1)



# ############################# Dynamic Variables #############################

prompt2 = PromptTemplate.from_template(
    """
Create a {days}-day travel plan for {destination}.

Budget: {budget}

Language: {language}
"""
)

final_prompt2 = prompt2.format(
    destination="Goa",
    days=3,
    budget="Medium",
    language="English"
)

print(final_prompt2)

# ############################# Chat Prompt Template #############################

from langchain_core.prompts import ChatPromptTemplate

prompt3 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly Telugu teacher."
    ),
    (
        "human",
        "Explain {topic}"
    )
])

messages3 = prompt3.format_messages(
    topic="Prompt Engineering"
)

print(messages3)

# ############################# Multi-Turn chat or Conversation ########################

from langchain_core.messages import HumanMessage

messages = [
    HumanMessage(
        content="Who is Virat Kohli?"
    ),
    HumanMessage(
        content="Where was he born?"
    )
]

# ########################### ChatPromptTemplate + History ###########################

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant."
    ),
    MessagesPlaceholder(
        variable_name="history"
    ),
    (
        "human",
        "{question}"
    )
])

history = [
    HumanMessage(
        content="Who is Virat Kohli?"
    )
]

messages = prompt.format_messages(
    history=history,
    question="Where was he born?"
)
print(messages)