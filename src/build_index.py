from pathlib import Path  # noqa: I001
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from exceptions import DocumentError, ValidationError
from logger import get_logger


logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SOURCE_FILE = DATA_DIR / 'faq_document.txt'


def load_faq_document() -> str:
    """Carga el documento de FAQs desde la carpeta data."""
    if not SOURCE_FILE.exists():
        raise DocumentError(f'No se encontró el archivo de datos: {SOURCE_FILE}')

    logger.info('Cargando documento de FAQs desde %s.', SOURCE_FILE)
    text = SOURCE_FILE.read_text(encoding='utf-8')
    logger.info('Documento de FAQs cargado (longitud: %s caracteres).', len(text))
    return text


def create_chunks(text: str) -> list[str]:
    """Divide el texto del documento en chunks."""
    splitter = CharacterTextSplitter(
        separator=' ',
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    logger.info('Se generaron %s chunks de texto.', len(chunks))
    return chunks


def generate_embeddings(chunks: list[str],) -> tuple[OpenAIEmbeddings, list[list[float]]]:
    """Genera embeddings para cada chunk y valida su cantidad."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    logger.info('Generando embeddings para %s chunks.', len(chunks))

    vectors = embeddings.embed_documents(chunks)
    if len(vectors) != len(chunks):
        raise ValidationError('Embeddings y chunks no coinciden.')

    logger.info('Se generaron %s embeddings.', len(vectors))
    return embeddings, vectors


def load_into_chroma(chunks: list[str], embeddings: OpenAIEmbeddings) -> Chroma:
    """Carga los chunks y embeddings en Chroma."""
    logger.info('Creando índice en Chroma con %s chunks.', len(chunks))
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name='faqs_hr',
    )
    logger.info('Índice guardado en Chroma correctamente.')
    return vectorstore


def build_index() -> Chroma:
    """Crea el índice en Chroma a partir del documento de FAQs."""
    load_dotenv()

    text = load_faq_document()
    chunks = create_chunks(text)
    embeddings, _ = generate_embeddings(chunks)
    vectorstore = load_into_chroma(chunks, embeddings)

    return vectorstore


if __name__ == '__main__':
    build_index()