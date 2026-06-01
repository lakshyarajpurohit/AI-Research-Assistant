"""
core/model_factory.py
Model-agnostic LLM factory.
Returns a LangChain chat model for any provider defined in settings.
Swapping models anywhere in the app = changing one string in the sidebar.
"""
from config.settings import MODELS, GROQ_API_KEY, GOOGLE_API_KEY


def get_llm(model_name: str, temperature: float = 0.2, streaming: bool = True):
    """
    Returns a LangChain BaseChatModel for the requested model_name.
    Falls back gracefully if an API key is missing.
    """
    config = MODELS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    provider = config["provider"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config["model_id"],
            api_key=GROQ_API_KEY,
            temperature=temperature,
            streaming=streaming,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config["model_id"],
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_model_info(model_name: str) -> dict:
    return MODELS.get(model_name, {})
