from io import BytesIO
import asyncio

import requests
from PIL import Image

from autogen_core.models import UserMessage
from autogen_core.models import ModelInfo
from autogen_core import Image as AGImage


from autogen_agentchat.messages import MultiModalMessage, StructuredMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient


pil_image = Image.open(BytesIO(requests.get("https://picsum.photos/300/200").content))
img = AGImage(pil_image)
multi_modal_message = MultiModalMessage(content=['Can you describe the content of this image?', img], source='User')

async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

async def web_search(query: str) -> str:
    """Find information on the web"""
    return f"AutoGen is a programming framework for building multi-agent applications."

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
    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[web_search],
        system_message="You are a helpful assistant. Use tools to solve tasks.",
        # reflect_on_tool_use=True,
        model_client_stream=True,
    )

    try:
        await Console(
            # agent.run_stream(task="What is the weather in New York")
            agent.run_stream(task=multi_modal_message),
            output_stats=True,
            # model_client.create_stream(messages=[multi_modal_message])
        )
    finally:
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(main())
