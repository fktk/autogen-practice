import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_agentchat.ui import Console


# Get the fetch tool from mcp-server-fetch.
fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

async def main():

    model_client = OpenAIChatCompletionClient(
        model='gemini-2.0-flash-lite',
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family='unknown',
            structured_output=True,
        ),
    )
    try:
        # Create an MCP workbench which provides a session to the mcp server.
        async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
            # Create an agent that can use the fetch tool.
            fetch_agent = AssistantAgent(
                name="fetcher", model_client=model_client, workbench=workbench, reflect_on_tool_use=True
            )

            await Console(
                fetch_agent.run_stream(task="Summarize the content of https://en.wikipedia.org/wiki/Seattle"),
                output_stats=True,
            )

    finally:
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(main())
