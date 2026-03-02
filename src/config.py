import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from exceptions import APIError
from logger import get_logger

logger = get_logger(__name__)


def init_llm() -> ChatOpenAI:
    """Inicializa el modelo de lenguaje."""
    logger.info("Inicializando modelo de lenguaje.")

    try:
        load_dotenv()

        # Inicializar clientes
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0003
        )
        logger.info("Modelo de lenguaje inicializado correctamente.")
    except Exception as e:
        raise APIError(f"Error al inicializar el modelo de lenguaje: {e}") from e
    return llm