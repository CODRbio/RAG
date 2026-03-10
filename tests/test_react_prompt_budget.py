from src.llm.react_loop import react_loop


class _FakeLlmClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, model=None, tools=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        return {
            "final_text": "ok",
            "reasoning_text": "",
            "tool_calls": [],
            "raw": {},
        }


def test_react_loop_applies_iteration_prompt_budget():
    client = _FakeLlmClient()
    huge = "x " * 40_000
    messages = [{"role": "system", "content": "system\n" + huge}]
    # Add many history turns to force budget trimming.
    for i in range(18):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"turn-{i} " + huge[:8000]})
    messages.append({"role": "user", "content": "final question"})

    result = react_loop(
        messages=messages,
        tools=[],
        llm_client=client,
        max_iterations=1,
        model="deepseek-chat",
        prompt_budget_min_output_tokens=4096,
        prompt_budget_safety_margin=0.1,
    )

    assert result.final_text == "ok"
    assert len(client.calls) == 1
    sent = client.calls[0]["messages"]
    # Budgeting keeps system + newest user and trims middle turns.
    assert len(sent) < len(messages)
    assert sent[0]["role"] == "system"
    assert sent[-1]["role"] == "user"
