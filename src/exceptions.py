"""Excepciones personalizadas para el sistema RAG.

Este módulo define una jerarquía de excepciones específicas del dominio
que facilitan el manejo de errores y proporcionan mensajes claros.
"""


class ErrorRAG(Exception):
    """Excepción base para todos los errores del sistema RAG.

    Todas las excepciones personalizadas heredan de esta clase base,
    permitiendo capturar cualquier error del sistema con un solo except.

    Examples:
        try:
            # código del sistema RAG
        except ErrorRAG as e:
            # captura cualquier error del sistema
    """

    pass


class APIError(ErrorRAG):
    """Error relacionado con la API externa (OpenAI).

    Se lanza cuando:
    - La API key no está configurada

    Examples:
        raise APIError("No se encontró OPENAI_API_KEY en las variables de entorno")
    """

    pass


class ValidationError(ErrorRAG):
    """Error de validación de datos o respuestas del modelo.

    Se lanza cuando:
    - La cantidad de embeddings generados no coincide con la cantidad de chunks

    Examples:
        raise ValidationError("La cantidad de embeddings generados no coincide con la cantidad de chunks")
    """

    pass


class InputError(ErrorRAG):
    """Error de entrada del usuario.

    Se lanza cuando:
    - El usuario no proporciona una consulta

    Examples:
        raise InputError("No se agregó la consulta. Usa --query 'tu consulta aquí'")
        raise InputError("La consulta no puede estar vacía")
    """

    pass


class DocumentError(ErrorRAG):
    """Error relacionado con la carga o procesamiento del documento fuente.

    Se lanza cuando:
    - El archivo no existe en la ruta indicada

    Examples:
        raise DocumentError("Archivo no encontrado: data/faq_document.txt")
    """

    pass
