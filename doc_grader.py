import os
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class GradeContext(BaseModel):
    """Binary score for relevance check on a retrieved document chunk."""
    binary_score: Literal["yes", "no"] = Field(
        description="Choose 'yes' if the document contains factual pieces relevant to the query. Otherwise 'no'."
    )
    rationale: str = Field(description="One-sentence explanation supporting the grading verdict.")

def create_doc_grader():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_grader = llm.with_structured_output(GradeContext)
    system_prompt = (
        "You are an objective compliance auditor grading text quality.\n"
        "Your sole task is to determine if the provided 'Retrieved Context' contains "
        "any information useful to answering the 'User Query'."
    )
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved Context:\n---\n{context}\n---\n\nUser Query: {query}")
    ])
    return grader_prompt | structured_grader

class AgentState(TypedDict):
    query: str
    documents: list[str]
    filtered_documents: list[str]

def doc_grader_node(state: AgentState) -> dict:
    print("--- EVALUATING RETRIEVED CONTEXT RELEVANCE ---")
    query = state["query"]
    documents = state["documents"]
    grader_chain = create_doc_grader()
    valid_documents = []
    
    for idx, doc in enumerate(documents):
        result: GradeContext = grader_chain.invoke({"query": query, "context": doc})
        print(f" -> Chunk [{idx + 1}] Evaluation: {result.binary_score.upper()}")
        if result.binary_score == "yes":
            valid_documents.append(doc)
            
    print(f"Filtering complete. Kept {len(valid_documents)} documents.")
    return {"filtered_documents": valid_documents}
