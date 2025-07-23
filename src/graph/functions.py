from copy import deepcopy
from typing import Dict, Any

from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage
from langchain_core.runnables import chain as as_runnable

from src.tools.answer_generator_tool import answer_generator_tool
from src.tools.query_qdrant_tool import query_qdrant_tool
from src.tools.subquery_generator_tool import subquery_generator_tool
from src.tools.tavily_search_tool import tavily_search_tool
from src.tools.topic_extractor_tool import topic_extractor_tool


@as_runnable
def run_topic_extractor_tool(state: Dict[Any, Any]) -> Dict[Any, Any]:
    result = deepcopy(state)
    user_query = result["messages"][-1].content
    tool_args = {
        "query": user_query,
    }
    topic_name = topic_extractor_tool.invoke(tool_args)
    action = AgentAction(
        tool="topic_extractor_tool",
        tool_input=tool_args,
        log=str(topic_name)
    )
    return {"messages": [AIMessage(str(topic_name))], "intermediate_steps": [(action, topic_name)], "topic_name": topic_name}


@as_runnable
def run_subquery_generator_tool(state: Dict[Any, Any]) -> Dict[Any, Any]:
    result = deepcopy(state)
    user_query = result["messages"][0].content
    tool_args = {
        "query": user_query,
    }
    subqueries = subquery_generator_tool.invoke(tool_args)
    action = AgentAction(
        tool="subquery_generator_tool",
        tool_input=tool_args,
        log=str(subqueries)
    )
    return {"messages": [AIMessage(str(subqueries))], "intermediate_steps": [(action, subqueries)], "subqueries": subqueries}


@as_runnable
def run_tavily_search_tool(state: Dict[Any, Any]) -> Dict[Any, Any]:
    result = deepcopy(state)
    user_query = result["query"]
    tool_args = {
        "query": user_query
    }
    search_result = tavily_search_tool.invoke(tool_args)
    action = AgentAction(
        tool="tavily_search_tool",
        tool_input={"query": user_query},
        log=search_result
    )

    return {"messages": [AIMessage(search_result)], "intermediate_steps": [(action, search_result)]}

@as_runnable
def run_document_retriever_tool(state: Dict[Any, Any]) -> Dict[Any, Any]:
    result = deepcopy(state)
    topic_name = result["topic_name"]
    query = result["query"]
    tool_args = {
        "query": query,
        "topic_name": topic_name.replace(" ", "_")
    }
    retrieved_docs = query_qdrant_tool.invoke(tool_args)


    return {"retrieved_docs": retrieved_docs}


@as_runnable
def run_answer_generator_tool(state: Dict[Any, Any]) -> Dict[Any, Any]:
    result = deepcopy(state)
    retrieved_docs = result["retrieved_docs"]
    query = result["messages"][0].content
    tool_args = {
        "query": query,
        "retrieved_data": retrieved_docs
    }
    answer = answer_generator_tool.invoke(tool_args)

    return {"final_answer": answer}