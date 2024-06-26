from dotenv import load_dotenv
load_dotenv()


"""
# Human-in-the-loop
In this example we will build a ReAct Agent that has a human in the loop. We will use the human to approve 
specific actions. This examples builds off the base chat executor. It is highly recommended you learn about that 
executor before going through this notebook. You can find documentation for that example here(https://github.com/
langchain-ai/langgraph/blob/66b728e83a0661d1ca8f47fad97ef53d3a5f717d/examples/chat_agent_executor_with_function_calling/base.ipynb. 
Any modifications of that example are called below with MODIFICATION, so if you are looking for the differences 
you can just search for that.

# Set up the tools
We will first define the tools we want to use. For this simple example, we will use create a placeholder search engine. 
However, it is really easy to create your own tools - see documentation here(https://python.langchain.com/v0.2/docs/how_to/custom_tools) 
on how to do that.
"""
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
"""

We can now wrap these tools in a simple ToolExecutor. This is a real simple class that takes in a ToolInvocation and calls that tool, 
returning the output. A ToolInvocation is any class with tool and tool_input attribute.
"""
from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)


"""
# Set up the model
Now we need to load the chat model we want to use. Importantly, this should satisfy two criteria:
   1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
   2. It should work with OpenAI function calling. This means it should either be an OpenAI model or a model that exposes a similar interface.
Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.
"""
from langchain_openai import ChatOpenAI

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True)

"""
After we've done this, we should make sure the model knows that it has these tools available to call. 
We can do this by converting the LangChain tools into the format for OpenAI function calling, and then bind them to the model class.
"""
model = model.bind_tools(tools)


"""
# Define the agent state
The main type of graph in langgraph is the StateGraph(https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph). 
This graph is parameterized by a state object that it passes around to each node. Each node then returns operations to update that state. 
These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute. 
Whether to set or add is denoted by annotating the state object you construct the graph with. For this example, the state we will track will 
just be a list of messages. We want each node to just add messages to that list. Therefore, we will use a TypedDict with one key (messages) and 
annotate it so that the messages attribute is always added to.
"""
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


"""
#Define the nodes
We now need to define a few different nodes in our graph. In langgraph, a node can be either a function or a 
runnable(https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel). There are two main nodes we need for this:
   1. The agent: responsible for deciding what (if any) actions to take.
   2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.
We will also need to define some edges. Some of these edges may be conditional. The reason they are conditional is that based on the output of a node, 
one of several paths may be taken. The path that is taken is not known until that node is run (the LLM decides).
   1. Conditional Edge: after the agent is called, we should either: a. If the agent said to take an action, 
      then the function to invoke tools should be called b. If the agent said that it was finished, then it should finish
   2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next
Let's define the nodes, as well as a function to decide how what conditional edge to take.
"""
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
    # We construct an ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    # We use the response to create tool messages
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # We return a list, because this will get added to the existing list
    return {"messages": tool_messages}

"""
# Define the graph
We can now put it all together and define the graph!
## MODIFICATION
We modify the graph to interrupt before calling the tools. This lets the user give approval to continue. 
Note that this is a simple example and we could modify it to change the tool input, use some other channel 
besides input, etc.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

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

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=MemorySaver(), interrupt_before=["action"])

"""
# Use it!
We can now use it! This now exposes the same interface as all other LangChain runnables.
"""
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in Cimahi?")]}
config = {"configurable": {"thread_id": "thread-1"}}
while True:
    for output in app.stream(inputs, config):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
    snapshot = app.get_state(config)
    # If "next" is present, it means we've interrupted mid-execution
    if not snapshot.next:
        break
    inputs = None
    response = input(
        "Do you approve the next step? Type y if you do, anything else to stop: "
    )
    if response != "y":
        break
