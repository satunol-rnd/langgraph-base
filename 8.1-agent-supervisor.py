from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Sequence, Optional
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


from pathlib import Path
_TEMP_DIRECTORY = Path.cwd() / "public" # TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "".join(lines[start:end])

@tool
def create_chart_image(
    csv_content: Annotated[str, "CSV content as a string."],
    file_name: Annotated[str, "File path to save the image."],
    title: Annotated[Optional[str], "The title of the chart. Default is 'Chart'."] = "Chart",
    summary: Annotated[Optional[str], "The summary of the chart. Default is ''."] = "",
) -> Annotated[str, "Path of the saved document file."]:
    """Create chart image and save it to disk."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from io import StringIO

        # Load the CSV content
        df = pd.read_csv(StringIO(csv_content))
        
        # Extract x and y axis labels from the first row
        x_label = df.columns[0]
        y_label = df.columns[1]

         # Extract data starting from the second row
        data = df.iloc[1:]

        # Separate the data into x and y, ensuring x can be any type (string or numeric)
        x_data = data.iloc[:, 0]
        y_data = data.iloc[:, 1].astype(float)

         # Create the plot
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        
        # Save the plot to a file
        plt.savefig(WORKING_DIRECTORY / "images" / file_name)
        plt.close()
    except Exception as e:
        return f"Failed to create chart image: {e}"
    return f"{summary}\nChart saved to images/{file_name}"


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

members = ["Researcher", "DocumentWriter", "DocumentReader", "ChartCreator", "Coder", "TextLenCalc", "AddAll"]
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

create_chart_image_agent = create_agent(
    llm,
    [create_chart_image],
    "You consume csv string data and generate a chart image.",
)
create_chart_image_node = functools.partial(agent_node, agent=create_chart_image_agent, name="ChartCreator")

research_agent = create_agent(
    llm, 
    [tavily_tool, write_document, create_chart_image], 
    "You are a web researcher. You may write documents to disk if asked. You may create charts if asked.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code.",
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

write_document_agent = create_agent(
    llm,
    [write_document],
    "You may write a document to disk.",
)
write_document_node = functools.partial(agent_node, agent=write_document_agent, name="DocumentWriter")

read_document_agent = create_agent(
    llm,
    [read_document, create_chart_image],
    "You may read a document from disk. You may also generate a chart from the csv document if asked.",
)
read_document_node = functools.partial(agent_node, agent=read_document_agent, name="DocumentReader")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("DocumentWriter", write_document_node)
workflow.add_node("DocumentReader", read_document_node)
workflow.add_node("ChartCreator", create_chart_image_node)
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

# To tell AI access our custom tools (TextLenCalc and AddAll)
# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content="Give me a list of total length of text in array: ['SATUNOL', 'BANDUNG', 'AI', 'SPECIALIST']. Then tell me the sum of the list.")
#         ]
#     },
#     {"recursion_limit": 12},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Summarize today weather in Bandung then write document with that summary as content and filename: 'weather_in_bandung.txt'"
#             )
#         ]
#     },
#     {"recursion_limit": 12},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Summarize today hourly weather in Cimahi "
                "then write document of csv with hour (in int) and temperature from that summary as content and filename: 'weather_in_cimahi_hourly.csv' "
                "finally, create chart image with that data with filename: 'weather_in_cimahi_hourly.png'"
            )
        ]
    },
    {"recursion_limit": 12},
):
    if "__end__" not in s:
        print(s)
        print("----")

# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Read document weather_in_bandung_hourly.csv and pass the output to create chart image with that data with filename: 'weather_in_bandung_hourly.png'"
#             )
#         ]
#     },
#     {"recursion_limit": 12},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Read document weather_in_bandung_hourly.csv and tell me the sumary of that data. You may also generate chart from it with filename: 'weather_in_bandung_hourly.png'."
#                 # content="Read document Bandung_Hourly_Weather_Forecast.txt and tell me the sumary of that data."
#             )
#         ]
#     },
#     {"recursion_limit": 12},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

# COUTION: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content="code to search file wich the name conatains '.env' in current working directory and tell me the content of the file.")
#         ]
#     },
#     {"recursion_limit": 12},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")
