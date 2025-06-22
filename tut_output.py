import asyncio

from typing import Literal
from pydantic import BaseModel

from autogen_core.models import ModelInfo
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
            system_message='Categorize the input as happy, sad or neutral following the JSON format.',
            output_content_type=AgentResponse,
        )

        await Console(
            agent.run_stream(task="I am great."),
            output_stats=True,
        )

    finally:
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(main())

