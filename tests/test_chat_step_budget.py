from src.api.routes_chat import _chat_effective_step_top_k


def test_chat_effective_step_top_k_applies_1p2_amplification():
    # ceil(60 * 1.2) = 72
    assert _chat_effective_step_top_k(60) == 72


def test_chat_effective_step_top_k_keeps_none_and_invalid():
    assert _chat_effective_step_top_k(None) is None
    assert _chat_effective_step_top_k(0) is None
    assert _chat_effective_step_top_k(-3) is None
