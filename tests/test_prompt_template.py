from src.server.prompt_template import build_messages


def test_builds_messages_with_system_prompt():
    chunks = [{"text": "The library closes at 10pm.", "id": "c1"}]
    messages = build_messages("When does the library close?", chunks)
    assert messages[0]["role"] == "system"
    assert "Ask Tupper" in messages[0]["content"]
    assert messages[-1]["role"] == "user"


def test_includes_context():
    chunks = [{"text": "Dining opens at 7am.", "id": "c1"}]
    messages = build_messages("What are the dining hours?", chunks)
    assert "Dining opens at 7am" in messages[0]["content"]


def test_includes_spotlighting():
    chunks = [{"text": "Test.", "id": "c1"}]
    messages = build_messages("q", chunks)
    assert "<<<" in messages[0]["content"]


def test_keeps_user_query_out_of_system_prompt():
    query = "UNIQUE_QUERY_12345"
    chunks = [{"text": "The library closes at 10pm.", "id": "c1"}]
    messages = build_messages(query, chunks)
    assert query not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": query}
