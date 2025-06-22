import asyncio

from typing import Literal
from pydantic import BaseModel

from autogen_core.models import ModelInfo
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


class AgentResponse(BaseModel):
    thoughts: str
    response: Literal['happy', 'sad', 'neutral']


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
        agent = AssistantAgent(
            name="assistant",
            model_client=model_client,
            system_message='You are a helpful assistant',
            model_context=BufferedChatCompletionContext(buffer_size=5),
        )

        await Console(
            agent.run_stream(task="My name is TK."),
            output_stats=True,
        )

        await Console(
            agent.run_stream(task="Who am I?"),
            output_stats=True,
        )

    finally:
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(main())

