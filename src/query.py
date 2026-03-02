from __future__ import annotations

import json
from typing import Any
import sys
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import argparse

from logger import get_logger
from build_index import build_index
from config import init_llm
from exceptions import InputError, ValidationError, APIError, DocumentError


logger = get_logger(__name__)
llm = init_llm()

PROMPT_TEMPLATE = """Responde la pregunta basandote UNICAMENTE
en el contexto proporcionado.
Si la informacion no esta en el contexto, di:
"No tengo informacion sobre eso en mi base de conocimiento".

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


class AnswerEvaluation(BaseModel):
    """Resultado estructurado de evaluación de respuesta RAG."""

    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=50)


def _format_docs(docs: list[Document]) -> str:
    return '\n\n'.join(doc.page_content for doc in docs)


def _build_vectorstore():
    """Construye el índice vectorial para la consulta actual."""
    return build_index()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Asistente para agentes de soporte al cliente"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help='Pregunta del usuario (ej: "Tengo problemas para entrar a mi cuenta")',
    )
    return parser.parse_args()



def evaluate_answer(
    user_question: str,
    system_answer: str,
    chunks_related: list[str],
) -> dict[str, Any]:
    """Evalúa respuesta RAG y devuelve score (0-10) + reason (>=50 chars)."""
    context_text = '\n---\n'.join(chunks_related)

    system_prompt = """Eres un evaluador de calidad de respuestas RAG.
    Debes devolver SIEMPRE:
    - score: entero entre 0 y 10
    - reason: texto en español de al menos 50 caracteres

    Evalúa al menos estas dimensiones:
    1) Relevancia de chunks: si los chunks se relacionan con la pregunta.
    2) Calidad de la respuesta: si la respuesta usa información de los chunks.
    3) Completitud: si cubre totalmente lo preguntado.

    Reglas:
    - Si detectas invención o contradicción con chunks, baja claramente el puntaje.
    - Justifica con observaciones concretas (menciona fortalezas y mejoras)."""

    user_msg = f"""Pregunta del usuario: {user_question}
    Respuesta del sistema: {system_answer}
    Chunks recuperados: {context_text}

    Devuelve score y reason cumpliendo las reglas."""
    logger.info('Evaluando respuesta con LLM.')

    llm_with_structure = llm.with_structured_output(AnswerEvaluation)
    result: AnswerEvaluation = llm_with_structure.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
    )
    logger.info('Respuesta evaluada correctamente.')
    return {
        'score': result.score,
        'reason': result.reason,
    }


def query_faq(k: int = 3) -> dict[str, Any]:
    """Ejecuta el flujo RAG y devuelve un JSON con respuesta y chunks."""
    args = parse_args()
    user_question = args.query
    if not user_question or not user_question.strip():
        raise InputError('La pregunta del usuario no puede estar vacia.')

    logger.info('Ejecutando flujo RAG con k=%s.', k)
    vectorstore = _build_vectorstore()
    retriever_k = vectorstore.as_retriever(search_kwargs={'k': k})
    retrieved_docs: list[Document] = retriever_k.invoke(user_question)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain_k = (
        {'context': retriever_k | _format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain_k.invoke(user_question)   
    chunks_related = [doc.page_content for doc in retrieved_docs]
    evaluation = evaluate_answer(user_question, answer, chunks_related)

    logger.info('Flujo RAG completado correctamente.')

    return {
        'user_question': user_question,
        'system_answer': answer,
        'chunks_related': chunks_related,
        'evaluation': evaluation,
    }


if __name__ == '__main__':
    try:
        result = query_faq()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except InputError as e:
        logger.error(f"Error de entrada: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Error de validación: {e}")
        sys.exit(1)
    except APIError as e:
        logger.error(f"Error de API: {e}")
        sys.exit(1)
    except DocumentError as e:
        logger.error(f"Error de documento: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado")
        sys.exit(1)








