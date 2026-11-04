import os
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    query: str
    transformed_query: str
    filtered_documents: list[str]

def create_query_rewriter():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "You are an AI optimization engine specializing in semantic search refinement.\n"
        "Your job is to take a failing user query and rewrite it to drastically improve "
        "vector database similarity matching.\n"
        "Output ONLY the raw optimized query string without quotes or introduction."
    )
    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Failing query to optimize: {query}")
    ])
    return rewriter_prompt | llm | StrOutputParser()

def query_rewriter_node(state: AgentState) -> dict:
    print("--- REWRITING FAILING USER QUERY ---")
    original_query = state["query"]
    rewriter_chain = create_query_rewriter()
    optimized_query = rewriter_chain.invoke({"query": original_query})
    optimized_query = optimized_query.strip().replace('"', '').replace("'", "")
    print(f" -> Optimized: \"{optimized_query}\"")
    return {"transformed_query": optimized_query}
