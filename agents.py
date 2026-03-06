from dotenv import load_dotenv
load_dotenv()

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# agent = create_react_agent(
#     model="groq:llama-3.3-70b-versatile",  
#     tools=[],  
#     prompt="You are a helpful assistant"  
# )

# Run the agent
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "who is modi"}]}
# )

# print(response["messages"][-1].content)
# checkpointer = InMemorySaver()


# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "when was he born"}]}
# )

# print(response["messages"][-1].content)





##Adding Memory
checkpointer = InMemorySaver()

agent = create_react_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[get_weather],
    checkpointer=checkpointer  
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "who is modi"}]},
    config  
)

config = {"configurable": {"thread_id": "2"}}

ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is his age"}]},
    config
)
print(sf_response["messages"][-1].content)
print(ny_response["messages"][-1].content)

try:
   img = agent.get_graph().draw_mermaid_png()
   with open("graph1.png", "wb") as f:
       f.write(img)
except Exception:
   pass



