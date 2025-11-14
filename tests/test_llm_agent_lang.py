import types

from agents.llm_agent import LLMAgent


class DummyLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        # mimic an object with .content attribute
        return types.SimpleNamespace(content='Esta es una respuesta de prueba.')


def test_llm_agent_respects_language(monkeypatch):
    dummy = DummyLLM()

    # Patch the get_llm function used by LLMAgent (it was imported directly)
    import agents.llm_agent as llm_agent_module

    monkeypatch.setattr(llm_agent_module, 'get_llm', lambda: dummy)

    # Prepare state with language set to Spanish
    state = {
        'conversation_history': [
            {'role': 'user', 'content': 'Tengo dolor de cabeza.'}
        ],
        'question': '¿Qué debo hacer para un fuerte dolor de cabeza?',
        'language': 'es'
    }

    out_state = LLMAgent(state)

    # Ensure LLM was invoked and prompt contains language instruction
    assert dummy.last_prompt is not None
    assert 'Answer in' in dummy.last_prompt or 'Answer in es' in dummy.last_prompt
    assert out_state.get('generation') is not None
