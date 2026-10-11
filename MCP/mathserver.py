
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiple(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# The transport="stdio" argument tells the server to use
# standard input/output (stdin and stdout) to receive and respond
# to tool function calls.
# STDIO: Used when the MCP client communicates with a local MCP server. You can test local tools,
#  and the client can display the results in your terminal.

if __name__ == "__main__":
    mcp.run(transport="stdio")
