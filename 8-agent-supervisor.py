from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Sequence
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

@tool
def get_text_length(
    text: Annotated[str, "The text to get the length of."],
) -> int:
    """Get the length of a text."""
    return len(text)

@tool
def add_all(
    numbers: Annotated[Sequence[int], "The numbers to add."],
) -> int:
    """Add all numbers together."""
    return sum(numbers)

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Create Agent Supervisor
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

members = ["Researcher", "Coder", "TextLenCalc", "AddAll"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# llm = ChatOpenAI(model="gpt-4-1106-preview")
llm = ChatOpenAI(model="gpt-3.5-turbo")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# Construct Graph
import functools
import operator
from typing import Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

text_length_agent = create_agent(
    llm,
    [get_text_length],
    "You may calculate the length of text.",
)
text_length_node = functools.partial(agent_node, agent=text_length_agent, name="TextLenCalc")

add_all_agent = create_agent(
    llm,
    [add_all],
    "You may add all numbers together.",
)
add_all_node = functools.partial(agent_node, agent=add_all_agent, name="AddAll")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("TextLenCalc", text_length_node)
workflow.add_node("AddAll", add_all_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content="Code to print 'Hello, LangGraph!' and write it into a txt file at"
#                          " current working directory + /public folder with the filename langgraph.txt.")
#         ]
#     }
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")


# for s in graph.stream(
#     {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
#     {"recursion_limit": 100},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Give me a list of total length of text in array: ['SATUNOL', 'AI', 'SPECIALIST']. Then tell me the sum of the list.")
        ]
    },
    {"recursion_limit": 12},
):
    if "__end__" not in s:
        print(s)
        print("----")
