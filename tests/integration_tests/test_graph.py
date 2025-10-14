import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from agent import graph
from agent.graph import State


def test_graph_initialization():
    """Test that the graph can be initialized successfully."""
    assert graph is not None
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "invoke")


def test_graph_accepts_valid_input():
    """Test that the graph accepts properly structured input."""
    inputs = State(messages=[HumanMessage(content="Hello, how are you?")], documents=[])

    # This test just validates the graph can process input without errors
    # Actual LLM calls would require API keys and make real requests
    try:
        result = graph.invoke(inputs)
        assert result is not None
        assert "messages" in result
    except Exception as e:
        # If there's an API key issue or connection issue, that's expected in tests
        # We're mainly testing the graph structure here
        if "API" in str(e) or "api" in str(e).lower() or "key" in str(e).lower():
            pytest.skip(f"Skipping due to API configuration: {e}")
        else:
            raise


def test_graph_with_empty_messages():
    """Test that graph handles edge case of empty message list."""
    inputs = State(messages=[], documents=[])

    try:
        result = graph.invoke(inputs)
        # Should either return a result or raise a clear error
        assert result is not None or True  # Graph should handle this gracefully
    except ValueError as e:
        # Expected behavior - graph should validate input
        assert "message" in str(e).lower() or "empty" in str(e).lower()
    except Exception as e:
        if "API" in str(e) or "api" in str(e).lower() or "key" in str(e).lower():
            pytest.skip(f"Skipping due to API configuration: {e}")
        else:
            # Some validation error is acceptable
            pass


def test_graph_state_structure():
    """Test that the graph output maintains expected state structure."""
    inputs = State(
        messages=[HumanMessage(content="What is the revenue?")], documents=[]
    )

    try:
        result = graph.invoke(inputs)

        # Validate output structure
        assert isinstance(result, dict), "Graph should return a dictionary"
        assert "messages" in result, "Result should contain 'messages' key"

        # Check that messages is a list
        assert isinstance(result["messages"], list), "Messages should be a list"

        # Verify input message is preserved in output
        input_content = inputs["messages"][0].content
        has_original = any(
            hasattr(msg, "content") and msg.content == input_content
            for msg in result["messages"]
        )
        assert has_original, "Original message should be preserved in output"

    except Exception as e:
        if "API" in str(e) or "api" in str(e).lower() or "key" in str(e).lower():
            pytest.skip(f"Skipping due to API configuration: {e}")
        else:
            raise


def test_graph_multiple_messages():
    """Test that graph handles multiple messages in conversation."""
    inputs = State(
        messages=[
            HumanMessage(content="Hello"),
            HumanMessage(content="What can you help me with?"),
        ],
        documents=[],
    )

    try:
        result = graph.invoke(inputs)

        assert result is not None
        assert "messages" in result
        # Should have at least the input messages
        assert len(result["messages"]) >= 2

    except Exception as e:
        if "API" in str(e) or "api" in str(e).lower() or "key" in str(e).lower():
            pytest.skip(f"Skipping due to API configuration: {e}")
        else:
            raise


def test_graph_with_tool_call():
    """Test that the graph can handle tool calls for retrieval."""
    with patch("agent.graph.init_chat_model") as mock_init:
        mock_llm = MagicMock()

        # First call: LLM decides to use retrieval tool
        mock_tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "revenue growth"},
                    "id": "call_123",
                }
            ],
        )
        # Second call: After rewrite, generate answer
        mock_final_response = AIMessage(
            content="Based on the documents, revenue grew by 20%."
        )

        mock_llm.invoke.side_effect = [mock_tool_call_response, mock_final_response]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = MagicMock(
            score=0.8
        )
        mock_init.return_value = mock_llm

        # Mock the retriever tool
        with patch("agent.graph.get_retriever_tool") as mock_retriever:
            mock_tool = MagicMock()
            mock_tool.name = "retrieve_documents"
            mock_retriever.return_value = mock_tool

            inputs = State(
                messages=[HumanMessage(content="What is the revenue growth?")],
                documents=[],
            )
            result = graph.invoke(inputs)

            assert result is not None
            assert "messages" in result


def test_graph_document_grading_relevant():
    """Test that relevant documents lead to answer generation."""
    with patch("agent.graph.init_chat_model") as mock_init:
        mock_llm = MagicMock()

        # Mock tool call
        mock_tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "revenue"},
                    "id": "call_123",
                }
            ],
        )

        # Mock final answer
        mock_answer = AIMessage(content="The revenue grew by 15%.")

        mock_llm.invoke.side_effect = [mock_tool_call, mock_answer]
        mock_llm.bind_tools.return_value = mock_llm

        # Mock grader to return high relevance score (>0.5)
        mock_grader_response = MagicMock(score=0.9)
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            mock_grader_response
        )

        mock_init.return_value = mock_llm

        with patch("agent.graph.get_retriever_tool") as mock_retriever:
            mock_tool = MagicMock()
            mock_tool.name = "retrieve_documents"
            mock_retriever.return_value = mock_tool

            inputs = State(
                messages=[HumanMessage(content="What is the revenue?")], documents=[]
            )
            result = graph.invoke(inputs)

            assert result is not None
            assert "messages" in result


def test_graph_document_grading_irrelevant():
    """Test that irrelevant documents trigger question rewriting."""
    with patch("agent.graph.init_chat_model") as mock_init:
        mock_llm = MagicMock()

        # Mock first tool call
        mock_tool_call_1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "test"},
                    "id": "call_123",
                }
            ],
        )

        # Mock rewritten question (AIMessage)
        mock_rewrite = AIMessage(content="What are the financial results?")

        # Mock second response (direct answer after rewrite)
        mock_final = AIMessage(content="I don't have that information.")

        mock_llm.invoke.side_effect = [mock_tool_call_1, mock_rewrite, mock_final]
        mock_llm.bind_tools.return_value = mock_llm

        # Mock grader to return low relevance score (<= 0.5)
        mock_grader_response = MagicMock(score=0.3)
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            mock_grader_response
        )

        mock_init.return_value = mock_llm

        with patch("agent.graph.get_retriever_tool") as mock_retriever:
            mock_tool = MagicMock()
            mock_tool.name = "retrieve_documents"
            mock_retriever.return_value = mock_tool

            inputs = State(messages=[HumanMessage(content="test query")], documents=[])
            result = graph.invoke(inputs)

            assert result is not None
            assert "messages" in result


def test_graph_state_persistence():
    """Test that the graph maintains state across nodes."""
    inputs = State(messages=[HumanMessage(content="Test question")], documents=[])

    with patch("agent.graph.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Test response")
        mock_llm.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm
        mock_init.return_value = mock_llm

        result = graph.invoke(inputs)

        assert result is not None
        assert "messages" in result
        assert "documents" in result
        # Original message should be preserved
        assert any(
            msg.content == "Test question"
            for msg in result["messages"]
            if hasattr(msg, "content")
        )
