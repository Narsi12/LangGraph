
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the weather location."""
    return "It's always raining in California"

# Streamable HTTP: The MCP client communicates with the MCP server over HTTP.
# Best for applications that need to access a server over a network.
#---------------------------------------------------------------------------------------------
# Used when the MCP client communicates with an MCP server over HTTP, either locally or remotely. 
# That server can call third-party APIs 
# and return their results, but STDIO servers can do that too.
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
