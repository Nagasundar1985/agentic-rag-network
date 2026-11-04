import os
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    query: str
    documents: list[str]
    route_to: str
    generation: str

class RouteDecision(BaseModel):
    """Decide where to route the user's initial query."""
    destination: Literal["vectorstore", "websearch"] = Field(
        description="Choose 'vectorstore' if the question relates to internal data. Choose 'websearch' for real-time facts."
    )
    rationale: str = Field(description="Brief reason for this routing decision.")

def router_node(state: AgentState) -> dict:
    print("--- ROUTING QUERY ---")
    query = state["query"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteDecision)
    
    system_prompt = (
        "You are an expert query router for an enterprise knowledge system.\n"
        "Your job is to analyze the user query and direct it to the correct datasource.\n"
        "Use 'vectorstore' for highly specific internal company data or financial filings.\n"
        "Use 'websearch' for current events, public news, or broader internet facts."
    )
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User query to route: {query}")
    ])
    router_chain = route_prompt | structured_llm_router
    decision: RouteDecision = router_chain.invoke({"query": query})
    print(f"Decision: Route to [{decision.destination}] | Reason: {decision.rationale}")
    return {"route_to": decision.destination}

def conditional_routing_edge(state: AgentState):
    return state["route_to"]
