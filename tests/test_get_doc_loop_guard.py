"""get_doc loop guard + decode-stop detector."""

from core.agent.parsing import history_has_get_doc_body, prompt_has_doc_body


def test_history_has_get_doc_body_detects_text():
    hist = [
        {"role": "toolresult", "name": "workspace.search", "content": '{"documents":[]}'},
        {
            "role": "toolresult",
            "name": "workspace.get_doc",
            "content": '{"doc_id": "abc", "text": "Quindlethorpe brass foundry"}',
        },
    ]
    assert history_has_get_doc_body(hist) is True


def test_history_has_get_doc_body_false_on_search_only():
    hist = [
        {"role": "toolresult", "name": "workspace.search", "content": '{"documents":[{"doc_id":"abc"}]}'},
    ]
    assert history_has_get_doc_body(hist) is False


def test_prompt_has_doc_body_on_get_doc_json():
    prompt = (
        '<myPT_toolresult>{"documents": [{"doc_id": "abc"}]}</myPT_toolresult>'
        '<myPT_assistant><myPT_toolcall>{"name": "workspace.get_doc"}</myPT_toolcall></myPT_assistant>'
        '<myPT_toolresult>{"doc_id": "abc", "text": "Quindlethorpe brass foundry"}</myPT_toolresult>'
        "<myPT_assistant>"
    )
    assert prompt_has_doc_body(prompt) is True


def test_prompt_has_doc_body_false_on_search_catalog():
    prompt = (
        '<myPT_toolresult>{"documents": [{"rank": 1, "doc_id": "abc", "snippet": "hi"}], "total": 1}</myPT_toolresult>'
        "<myPT_assistant>"
    )
    assert prompt_has_doc_body(prompt) is False
