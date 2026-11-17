"""Selection may be stochastic; evidence identity is checked deterministically."""

import json
from types import SimpleNamespace

import pytest

from llm_toolkit.agents.memory import compress_dialogue, render_memory

RECORDS = [
    {"id":"p0","speaker":"Ada","audience":"public","content":"We might inspect; nobody has agreed."},
    {"id":"p1","speaker":"Ben","audience":"public","content":'Ada said "we", not "Ben will".'},
]


def run(response, **options):
    owner = SimpleNamespace(model="offline", completion_options={"num_retries":0}, token_usage={"total_tokens":0})
    class Selector:
        def __init__(self, **kwargs):
            self.messages=[]
            self.token_usage={"total_tokens":5}
            assert kwargs["completion_options"] == {"num_retries":0}
        def send_text(self, text):
            pass
        def get_response(self, **kwargs):
            self.messages.append({"content":response})
    return compress_dialogue(owner, RECORDS, summarizer_factory=Selector, keep_recent=options.pop("keep_recent", 0), **options)


def test_quotes_are_rendered_from_sources_in_source_order():
    memory=run('{"source_ids":["p1","p0"]}')
    assert memory["records"] == RECORDS
    assert memory["text"] == render_memory(RECORDS)
    assert memory["source_count"] == memory["selected_count"] == 2


@pytest.mark.parametrize("response", [
    '{"source_ids":["made-up"]}', '{"source_ids":["p0","p0"]}',
    '{"source_ids":[1]}', '{"source_ids":"p0"}',
    '{"source_ids":["p0"],"summary":"Ben will inspect"}', 'Ben will inspect', 'null',
])
def test_invalid_or_rewritten_memory_is_rejected(response):
    with pytest.raises(ValueError):
        run(response)


def test_both_record_and_character_limits_are_enforced():
    with pytest.raises(ValueError, match="record limit"):
        run('{"source_ids":["p0","p1"]}', max_records=1)
    with pytest.raises(ValueError, match="character budget"):
        run('{"source_ids":["p0"]}', max_characters=len(render_memory([])))


def test_recent_records_are_retained_even_when_model_selects_nothing():
    memory = run('{"source_ids":[]}', keep_recent=4)
    assert memory["records"] == RECORDS
    assert memory["required_ids"] == ["p0", "p1"]


def test_recent_floor_cannot_silently_break_the_total_budget():
    with pytest.raises(ValueError, match="record limit"):
        run('{"source_ids":[]}', keep_recent=4, max_records=1)
