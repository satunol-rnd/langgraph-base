from dotenv import load_dotenv
load_dotenv()

# more at : https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

_TEMP_DIRECTORY = Path.cwd() / "public" # TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)
PDF_DIRECTORY = WORKING_DIRECTORY / "pdf"

def initialize_vectorstore(embeddings):
    folder = "index"
    index_path = Path.cwd() / folder
    if not os.path.exists(index_path):
        print("Creating FAISS index")
        all_documents = []
        
        # Load all PDF files in the directory
        for pdf_file in PDF_DIRECTORY.glob("*.pdf"):
            try:
                print(f"Loading {pdf_file.name}")
                loader = PyPDFLoader(file_path=pdf_file)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        # Split documents
        text_splitter = CharacterTextSplitter(
            chunk_size=300, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=all_documents)

        # Create and save the vectorstore
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        print("FAISS index created")
    
    # Load and return the vectorstore
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Add to vectorDB
emmbeddings = OpenAIEmbeddings()
vectorstore = initialize_vectorstore(emmbeddings)
retriever = vectorstore.as_retriever()

# Create tool
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Mental Health.",
)

tools = [retriever_tool]

#Agent State
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import tools_condition


### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    # Define the prompt template
    prompt_template_str = (
        "You are a helpful AI assistant named Tifa that helps human to retrieve documents based on their questions. "
        " If there is no relevant document, respond with 'sorry, there is no relevant document'. \n"
        "{messages}"
    )
    # Create a PromptTemplate instance
    prompt_template = PromptTemplate(
        input_variables=["messages"],
        template=prompt_template_str
    )
    
    # Extract messages from the state
    messages = state["messages"]
    
    # Initialize the model (uncomment the desired model line)
    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    model = model.bind_tools(tools)
    
    # Format the messages using the prompt template
    formatted_prompt = prompt_template.format(messages=messages)
    
    # Invoke the model with the formatted prompt
    response = model.invoke(formatted_prompt)
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


from langchain_core.prompts import ChatPromptTemplate
# Prompt
prompttemplate = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use less than 300 words and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template=prompttemplate)

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Graph
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.set_entry_point("agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

# inputs = {
#     "messages": [
#         ("user", "how to cure PTSD?"),
#     ]
# }
# for output in graph.stream(inputs, {"recursion_limit": 6}):
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value, indent=2, width=80, depth=None)
#     print("\n---\n")

# inputs = {
#     "messages": [
#         ("user", "hi, my Name is Roy. What is your name?"),
#     ]
# }
# for output in graph.stream(inputs, {"recursion_limit": 6}):
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")

# inputs = {
#     "messages": [
#         ("user", "Tifa, tolong jelaskan apa itu Post Traumatic Stress Disorder (PTSD)?"),
#     ]
# }
# for output in graph.stream(inputs, {"recursion_limit": 6}):
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")

inputs = {
    "messages": [
        ("user", "Tifa, tolong jelaskan microprocessor?"),
    ]
}
for output in graph.stream(inputs, {"recursion_limit": 6}):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
