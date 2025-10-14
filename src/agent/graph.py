"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Sequence
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from tools import get_retriever_tool
from utils import get_latest_messages


class State(MessagesState):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    documents: list[str]


class ResponseAgent:
    def __init__(
        self,
        model_name: str = "openai:gpt-5-nano",
        tools: Sequence[BaseTool] | None = None,
    ):
        self.model_name: str = model_name
        self.tools: Sequence[BaseTool] = tools or []
        self.llm = None

    def _get_llm(self):
        """Lazy initialization of the LLM to avoid repeated setups."""
        if self.llm is None:
            self.llm = init_chat_model(self.model_name, temperature=0)
            if self.tools:
                self.llm = self.llm.bind_tools(self.tools)
        return self.llm

    def generate_query_or_respond(self, state: State) -> Dict[str, Any]:
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
        """
        llm = self._get_llm()
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def rewrite_question(self, state: State) -> Dict[str, Any]:
        """Rewrite the original user question."""
        REWRITE_PROMPT = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        # Get the most recent user message
        question, _ = get_latest_messages(state, return_context=False)

        prompt = REWRITE_PROMPT.format(question=question)
        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}

    def generate_answer(self, state: MessagesState) -> Dict[str, Any]:
        """Generate an answer."""
        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise. "
            "Include the source document name and page number of the context used to form your answer.\n"
            "Question: {question} \n"
            "Context: {context}"
        )
        # Get the most recent user message and tool context
        question, context = get_latest_messages(state, return_context=True)

        prompt = GENERATE_PROMPT.format(question=question, context=context)
        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}


class GradeDocuments(BaseModel):
    """Grade documents using a score for relevance check."""

    score: float = Field(
        description="Relevance score: a float value representing relevance from 0 to 1"
    )


class GraderAgent:
    def __init__(self, model_name: str = "openai:gpt-5-nano"):
        self.model_name: str = model_name
        self.llm = None

    def _get_llm(self):
        """Lazy initialization of the LLM to avoid repeated setups."""
        if self.llm is None:
            self.llm = init_chat_model(self.model_name, temperature=0)
        return self.llm

    def grade_documents(
        self, state: MessagesState
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        GRADE_PROMPT = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a score from 0 to 1 to indicate the relevance of the document to the question."
        )
        # Get the most recent user message and tool context
        question, context = get_latest_messages(state, return_context=True)

        prompt = GRADE_PROMPT.format(question=question, context=context)
        llm = self._get_llm()
        response = llm.with_structured_output(GradeDocuments).invoke(
            [HumanMessage(content=prompt)]
        )
        score = getattr(response, "score", 0)

        if score > 0.5:
            return "generate_answer"
        else:
            return "rewrite_question"


tools = [get_retriever_tool()]

# Initialize agents
response_agent = ResponseAgent(tools=tools)
grader_agent = GraderAgent()

# Define the nodes we will cycle between
workflow = StateGraph(State)
workflow.add_node(response_agent.generate_query_or_respond)
workflow.add_node("retrieve", ToolNode(tools=tools))
workflow.add_node(response_agent.rewrite_question)
workflow.add_node(response_agent.generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Assess document relevance
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grader_agent.grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
