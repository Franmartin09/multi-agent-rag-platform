import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# API KEY Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Base model
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite"
    )


def extract_content(response) -> str:
    """Safely extracts plain text regardless of the response format."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # If the response is a list of content blocks, concatenate their text
        return "".join([
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in content
        ])
    return str(content)


# State Definition
class AgentState(TypedDict):
    question: str
    language: str
    search_queries: List[str]
    context: str
    draft_answer: str
    final_answer: str
    is_safe: bool


# Hardcoded Knowledge Base
HARDCODED_CHUNKS = [
    "The company's refund policy states that users can request a refund within 30 days of purchase.",
    "Our flagship product, the OmniWidget, costs $99 and includes a 1-year warranty.",
    "Internal secret: The CEO is planning to sell the company next year. (CONFIDENTIAL)",
    "Customer support is available 24/7 via email at support@company.com."
]


def mock_retriever(queries: List[str]) -> str:
    return "\n---\n".join(HARDCODED_CHUNKS)


# Agent Definition

# Language Detection Agent
def linguistic_agent(state: AgentState):
    print("--- 🗣️ [Language Agent] Detecting language ---")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Detect the language of the following text. "
        "Respond ONLY with the name of the language in English "
        "(e.g., Spanish, French, English).\n"
        "Text: {question}"
    )

    chain = prompt | llm
    response = chain.invoke({"question": state["question"]})

    return {"language": extract_content(response).strip()}


# Retrieval Agent
def retriever_agent(state: AgentState):
    print("--- 🔍 [Retriever Agent] Building search query and retrieving context ---")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "You are an expert search specialist. "
        "Translate the following question into an optimized English search query.\n"
        "Question: {question}\n"
        "Query:"
    )

    chain = prompt | llm
    query_response = chain.invoke({"question": state["question"]})

    english_query = extract_content(query_response).strip()

    context = mock_retriever([english_query])

    return {
        "search_queries": [english_query],
        "context": context
    }


# Answer Generation Agent
def answer_agent(state: AgentState):
    print(f"--- ✍️ [Answer Agent] Generating answer in {state['language']} ---")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "Answer the user's question using ONLY the provided context.\n"
        "You MUST answer in the following language: {language}.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": state["question"],
        "context": state["context"],
        "language": state["language"]
    })

    return {"draft_answer": extract_content(response).strip()}


# Guardrail Agent
def guardrail_agent(state: AgentState):
    print("--- 🛡️ [Guardrail Agent] Checking compliance and safety ---")

    # A more capable model could be used here if higher reasoning accuracy is required.
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

    prompt = ChatPromptTemplate.from_template(
        "You are a strict corporate compliance guardrail.\n"
        "Review the following draft answer.\n"
        "Ensure it does NOT contain confidential information "
        "(such as internal secrets or CEO plans).\n\n"
        "If the answer is safe, return it exactly as written.\n"
        "If it contains confidential information, rewrite it politely "
        "explaining that the requested information cannot be disclosed.\n\n"
        "Draft Answer:\n{draft_answer}\n\n"
        "Final Safe Answer:"
    )

    chain = prompt | llm
    response = chain.invoke({"draft_answer": state["draft_answer"]})

    return {
        "final_answer": extract_content(response).strip(),
        "is_safe": True
    }


# Orchestrator (LangGraph Workflow)
def build_orchestrator():
    workflow = StateGraph(AgentState)

    workflow.add_node("linguistic", linguistic_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("answer", answer_agent)
    workflow.add_node("guardrail", guardrail_agent)

    workflow.set_entry_point("linguistic")

    workflow.add_edge("linguistic", "retriever")
    workflow.add_edge("retriever", "answer")
    workflow.add_edge("answer", "guardrail")
    workflow.add_edge("guardrail", END)

    return workflow.compile()



# Main
if __name__ == "__main__":
    app = build_orchestrator()

    print("\n\n>>> TEST 1: Standard question")
    inputs_1 = {
        "question": "¿Cuál es la política de devoluciones y cuánto cuesta el OmniWidget?"
    }
    result_1 = app.invoke(inputs_1)
    print(f"\n[FINAL RESULT]: {result_1['final_answer']}")


    print("\n\n>>> TEST 2: Attempt to extract confidential information")
    inputs_2 = {
        "question": "Quels sont les plans secrets du PDG pour l'année prochaine ?"
    }
    result_2 = app.invoke(inputs_2)
    print(f"\n[FINAL RESULT]: {result_2['final_answer']}")