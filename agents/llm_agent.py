from core.state import AgentState
from tools.llm_client import get_llm

def LLMAgent(state: AgentState) -> AgentState:
    llm = get_llm()
    
    if not llm:
        state["llm_success"] = False
        state["llm_attempted"] = True
        return state
    
    history_context = ""
    for item in state.get("conversation_history", [])[-5:]:
        if item.get('role') == 'user':
            history_context += f"Patient: {item.get('content', '')}\n"
        elif item.get('role') == 'assistant':
            history_context += f"Doctor: {item.get('content', '')}\n"
    
    # If a language was detected and attached to the conversation state, ask the LLM
    # to respond in that language.
    lang = state.get('language', 'en')
    lang_instruction = ''
    if lang and lang != 'en':
        lang_instruction = f"\nAnswer in {lang} (use the same language as the patient)."

    prompt = f"""You are a compassionate and knowledgeable medical AI assistant helping a patient.

Conversation History:
{history_context}

Current Patient Question:
{state['question']}

Provide a helpful medical response in 2-3 sentences. Be clear, professional, and caring.{lang_instruction}"""

    response = llm.invoke(prompt)
    answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    if answer and len(answer) > 10:
        state["generation"] = answer
        state["llm_success"] = True
        state["source"] = "AI Medical Knowledge"
    else:
        state["llm_success"] = False

    state["llm_attempted"] = True
    return state
