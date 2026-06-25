# A toy example to use claude-agent-sdk. Run it inside claude
# to make sure authentication pass.

import anyio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
