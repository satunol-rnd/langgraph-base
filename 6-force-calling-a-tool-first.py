from dotenv import load_dotenv
load_dotenv()

# complete tutorial : https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/force-calling-a-tool-first.ipynb

from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]


from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)


from langchain_openai import ChatOpenAI

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True)

model = model.bind_tools(tools)


import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


"""
## MODIFICATION
Here we create a node that returns an AIMessage with a tool call 
- we will use this at the start to force it call a tool
"""
# This is the new first - the first call of the model we want to explicitly hard-code some action
from langchain_core.messages import AIMessage


def first_model(state: AgentState):
    human_input = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_search_results_json",
                        "args": {
                            "query": human_input,
                        },
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(AgentState)

# Modification - Define the new entrypoint
workflow.add_node("first_agent", first_model)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("first_agent") # Modification - we now use `first_agent`

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Modification - After we call the first agent, we know we want to go to action
workflow.add_edge("first_agent", "action")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


from langchain_core.messages import HumanMessage

# inputs = {"messages": [HumanMessage(content="what is the weather in cimahi?")]}
# inputs = {"messages": [HumanMessage(content="who is Joko Widodo?")]}
inputs = {"messages": [HumanMessage(content="tell me about the weather in Denpasar?")]}

# app.invoke(inputs)
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")