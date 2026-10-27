from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_toolkit.agents.agent import Agent
from llm_toolkit.agents.director import Director


def response(content=None, **extra):
    return SimpleNamespace(choices=[SimpleNamespace(message={"role": "assistant", "content": content, **extra})],
                           usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})


@pytest.fixture
def director():
    with patch.object(Agent, "_update_model_support"):
        cast = {name: Agent() for name in ("Ada", "Lin")}
        coordinator = Agent()
    for agent in [*cast.values(), coordinator]:
        agent.send_text("Existing history")
    return Director(cast, coordinator)


def test_failed_second_reply_restores_all_histories_and_metadata_but_retains_usage(director):
    agents = [*director.agents.values(), director.director_agent]
    originals = [agent.messages for agent in agents]
    before = [deepcopy(agent.messages) for agent in agents]
    director.emotions = {"Ada": "neutral"}
    director.reply_metadata = [{"speaker": "Ada", "emotion": "neutral"}]
    with patch("llm_toolkit.agents.agent.completion", side_effect=[
        response('{"speaker":"Ada","emotion":"happy"}'), response("First reply"),
        response('{"speaker":"Lin"}'), RuntimeError("second reply failed"),
    ]), pytest.raises(RuntimeError, match="second reply"):
        director.public_turn("Question")
    for agent, original, expected in zip(agents, originals, before):
        assert agent.messages is original
        assert agent.messages == expected
    assert director.public_transcript == []
    assert director.emotions == {"Ada": "neutral"}
    assert director.reply_metadata == [{"speaker": "Ada", "emotion": "neutral"}]
    assert len(director.decisions) == 2
    assert director.director_agent.token_usage["total_tokens"] == 4
    assert director.agents["Ada"].token_usage["total_tokens"] == 2


@pytest.mark.parametrize("isolate", [False, True])
def test_failed_private_turn_restores_history_and_transcript(director, isolate):
    before = deepcopy(director.agents["Ada"].messages)
    with patch("llm_toolkit.agents.agent.completion", side_effect=RuntimeError("failed")), pytest.raises(RuntimeError):
        director.private_turn("Ada", "secret", isolate=isolate)
    assert director.agents["Ada"].messages == before
    assert director.private_transcripts == {}
    assert director._private_messages == {}


def test_isolated_private_history_persists_without_entering_public_requests(director):
    with patch("llm_toolkit.agents.agent.completion", side_effect=[
        response("Secret answer"), response("Another secret answer"),
        response('{"speaker":"Ada"}'), response("Public answer"),
    ]) as completion:
        director.private_turn("Ada", "Secret input", isolate=True)
        director.private_turn("Ada", "Follow up", isolate=True)
        private_request = deepcopy(completion.call_args.kwargs["messages"])
        director.public_turn("Public question", ai_turns=1)
        public_request = deepcopy(completion.call_args.kwargs["messages"])
    assert "Secret input" in str(private_request)
    assert "Secret answer" in str(private_request)
    assert "Secret" not in str(public_request)
    assert "Secret" not in str(director.agents["Lin"].messages)
    assert len(director.get_private_transcript("Ada")) == 4


def test_tool_exhaustion_is_rejected_locally_until_results_are_supplied():
    call = {"id": "call_1", "type": "function", "function": {"name": "work", "arguments": "{}"}}
    with patch.object(Agent, "_update_model_support"):
        agent = Agent()
    agent.tool_handlers["work"] = lambda: "done"
    with patch("llm_toolkit.agents.agent.completion", return_value=response(tool_calls=[call])):
        agent.get_response(max_tool_rounds=0)
    before = deepcopy(agent.messages)
    with patch("llm_toolkit.agents.agent.completion") as completion, pytest.raises(RuntimeError, match="Unresolved tool calls"):
        agent.get_response(context="temporary")
    completion.assert_not_called()
    assert agent.messages == before
    agent.messages.append({"role": "tool", "tool_call_id": "call_1", "content": "done"})
    with patch("llm_toolkit.agents.agent.completion", return_value=response("Finished")):
        assert agent.get_response().choices[0].message["content"] == "Finished"
