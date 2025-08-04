from dotenv import load_dotenv
load_dotenv()
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import os


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def run_agent():
   client = MultiServerMCPClient(
       {
           "github": {
               "command": "npx",
               "args": [
                   "-y",
                   "@modelcontextprotocol/server-github"
               ],
               "env": {
                   "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
               },
               "transport": "stdio"
           },
           "filesystem": {
               "command": "npx",
               "args": [
                   "-y",
                   "@modelcontextprotocol/server-filesystem",
                   "/Users/keertipurswani/GitHub/GenAI Bootcamp"
               ],
               "transport": "stdio"
           }
       }
   )
   tools = await client.get_tools()
   print("Tools -----------------")
   print(tools)
   print("-----------------------------")
   agent = create_react_agent("openai:gpt-4o", tools)
   response = await agent.ainvoke({"messages": "create an empty file educosys.txt in the directory D:/4_Technology/KP_Gen_AI_Bootcamp/KP-Gen-AI-Parent-Folder"})
   print(response["messages"][-1].content)


if __name__ == "__main__":
   asyncio.run(run_agent())