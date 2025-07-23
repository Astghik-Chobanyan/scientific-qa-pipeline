import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from loguru import logger

from src.graph.functions import run_topic_extractor_tool, run_document_retriever_tool, run_answer_generator_tool, \
    run_subquery_generator_tool, run_tavily_search_tool

load_dotenv()

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    topic_name: str
    subqueries: List[str]
    retrieved_docs: Annotated[List[str], operator.add]
    final_answer: str


TOPICS = {
    "Diffusion Models in Computer Vision": "Diffusion_Models_in_Computer_Vision",
    "Graph Neural Networks (GNNs) for Molecular Property Prediction": "Graph_Neural_Networks_(GNNs)_for_Molecular_Property_Prediction",
    "Transformer Models for Protein Folding": "Transformer_Models_for_Protein_Folding",
    "Large Language Models for Mathematical Reasoning": "Large_Language_Models_for_Mathematical_Reasoning"
}


def continue_to_run_document_retriever_tool(state):
    topic_name = state["topic_name"]
    subqueries = state["subqueries"]
    tavily_search_states = []
    document_retrievers_states = []
    if topic_name not in TOPICS:
        logger.warning(f"Topic '{topic_name}' not found in predefined topics. Will continue with Tavily search.")
        if not subqueries:
            tavily_search_states.append({"query": state["messages"][0].content})
        for subquery in subqueries:
            tavily_search_states.append({"query": subquery})
        return [Send("tavily_search_node", tavily_search_state) for tavily_search_state in tavily_search_states]
    else:
        for subquery in subqueries:
            document_retrievers_states.append({"query": subquery, "topic_name": topic_name})
        return [Send("retrieve_documents_node", document_retriever_states) for document_retriever_states in document_retrievers_states]



class ResearchQA:
    def __init__(self):
        self.workflow = StateGraph(GraphState)

        # Add nodes
        self.workflow.add_node("extract_topic", run_topic_extractor_tool)
        self.workflow.add_node("subquery_generator", run_subquery_generator_tool)
        self.workflow.add_node("tavily_search_node", run_tavily_search_tool)
        self.workflow.add_node("retrieve_documents_node", run_document_retriever_tool)
        self.workflow.add_node("generate_answer", run_answer_generator_tool)

        # Add edges
        self.workflow.set_entry_point("extract_topic")
        self.workflow.add_edge("extract_topic", "subquery_generator")
        self.workflow.add_conditional_edges("subquery_generator", continue_to_run_document_retriever_tool)
        self.workflow.add_edge("tavily_search_node", "generate_answer")
        self.workflow.add_edge("retrieve_documents_node", "generate_answer")
        self.workflow.add_edge("generate_answer", END)

        self.graph = self.workflow.compile(checkpointer=MemorySaver())


if __name__ == '__main__':
    graph = ResearchQA().graph
    config = {"configurable": {"thread_id": '111'}}
    state = graph.get_state(config)
    query = "How AnyI2V enables motion-controlled video generation using a training-free approach."
    # query = "What are the recent advancements in quantum error correction techniques for fault-tolerant quantum computing?"
    if "messages" not in state.values:
        state.values["messages"] = []
    state.values["messages"].append(HumanMessage(query))
    for event in graph.stream(state.values, config):
        response = event
    answer = response["generate_answer"]["final_answer"]
    logger.success(f"Query: {query} \nAnswer: {answer}")