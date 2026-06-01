"""
utils/cost_tracker.py
Real-time token usage and cost estimation.
Tracks every LLM call and accumulates session spend.
"""
import tiktoken
from config.settings import MODELS


class CostTracker:
    def __init__(self):
        self.session_calls:  list = []
        self.total_input_tokens:  int   = 0
        self.total_output_tokens: int   = 0
        self.total_cost_usd:      float = 0.0

    def estimate_tokens(self, text: str) -> int:
        """Approximate token count using cl100k (close enough for all models)."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4   # fallback: ~4 chars/token

    def log_call(self, model_name: str, input_text: str, output_text: str) -> dict:
        model_info    = MODELS.get(model_name, {})
        input_tokens  = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)

        cost_in  = (input_tokens  / 1_000_000) * model_info.get("cost_per_1m_input",  0)
        cost_out = (output_tokens / 1_000_000) * model_info.get("cost_per_1m_output", 0)
        call_cost = cost_in + cost_out

        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd      += call_cost

        call_record = {
            "model":          model_name,
            "input_tokens":   input_tokens,
            "output_tokens":  output_tokens,
            "cost_usd":       call_cost,
        }
        self.session_calls.append(call_record)
        return call_record

    def get_summary(self) -> dict:
        return {
            "total_calls":         len(self.session_calls),
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "calls":               self.session_calls,
        }

    def reset(self):
        self.__init__()


# Session-level singleton (used across Streamlit reruns via st.session_state)
def get_tracker() -> CostTracker:
    return CostTracker()
