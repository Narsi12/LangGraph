
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

load_dotenv()

# we have two servers one is mathserver and wather
async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],  # Ensure correct file path
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )
    import os
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    tools = await client.get_tools()
    model = ChatGroq(model="openai/gpt-oss-120b")
    agent = create_react_agent(
        model,tools
    )
    math_response = await agent.ainvoke(
        {"messages":[{"role":"user","content":"What's (3+5)x 12?"}]}
    )
    print("maths response:",math_response["messages"][-1].content)
    ## keep weather server running ok then only code will execute
    weather_response = await agent.ainvoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in california ?"}
        ]
    }
    )
    print("Weather response:", weather_response["messages"][-1].content)

asyncio.run(main())
 
