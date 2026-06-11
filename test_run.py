from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "research_agent:v1"
)

messages = prompt.invoke(
    {
        "topic": "Compare YOLO v8 vs DETR for object detection"
    }
)

print(messages)

# result = agent.invoke(
#     {
#         "messages": messages
#     }
# )