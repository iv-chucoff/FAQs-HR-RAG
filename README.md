# FAQs-HR-RAG

Sistema de preguntas frecuentes para HR basado en RAG (Retrieval-Augmented Generation).  
Recupera fragmentos relevantes de un documento interno de FAQs y genera respuestas automáticas para consultas de clientes.

## Proposito del proyecto

Este proyecto nace para automatizar respuestas a preguntas frecuentes de clientes en un contexto de HR, usando una base de conocimiento ya existente en formato documento.

En lugar de responder solo con conocimiento general del modelo, el flujo:

- busca informacion en el documento fuente de FAQs;
- recupera los fragmentos mas parecidos a la consulta;
- genera una respuesta limitada a ese contexto;
- y luego evalua la calidad de esa respuesta respecto a los chunks recuperados.

El objetivo principal es reducir tiempos de respuesta y mejorar consistencia, evitando respuestas inventadas cuando la informacion no existe en la base de conocimiento.

## Arquitectura del proyecto

```text
FAQs-HR-RAG/
├─ data/
│  └─ faq_document.txt         # Documento fuente con las FAQs
├─ outputs/
│  └─ sample_queries.json      # 3 ejemplos de salida del sistema
├─ src/
│  ├─ build_index.py           # Chunking + embeddings + carga en Chroma
│  ├─ config.py                # Configuracion del modelo OpenAI
│  ├─ exceptions.py            # Excepciones personalizadas
│  ├─ logger.py                # Logger con salida coloreada
│  └─ query.py                 # Flujo principal: retrieval + respuesta + evaluacion
├─ .env.example
├─ pyproject.toml
└─ uv.lock
```

### Rol de cada modulo

- `data/faq_document.txt`: contiene el documento fuente con las FAQs de negocio.
- `outputs/sample_queries.json`: contiene ejemplos reales de respuestas devueltas por el pipeline.
- `src/build_index.py`: carga el documento, genera chunks, calcula embeddings y crea el indice vectorial en Chroma.
- `src/config.py`: inicializa el modelo de OpenAI usado para generar respuestas y para evaluar.
- `src/exceptions.py`: define errores de dominio (entrada, validacion, API, documento).
- `src/logger.py`: centraliza logs con formato consistente y colores por nivel.
- `src/query.py`: recibe la consulta del cliente, recupera contexto relevante, genera respuesta y ejecuta evaluacion de calidad.

## Flujo RAG end-to-end

1. Se carga el documento de FAQs (`data/faq_document.txt`).
2. Se divide en chunks de texto.
3. Se generan embeddings para cada chunk.
4. Se indexa en Chroma para busqueda vectorial.
5. Llega la consulta del usuario por CLI (`--query`).
6. Se recuperan los `k` chunks mas relevantes.
7. El LLM responde usando solo ese contexto.
8. Un segundo paso evalua la respuesta (`score` y `reason`) en funcion de los chunks recuperados.

## Decisiones tecnicas

### Estrategia de chunking

Se uso `CharacterTextSplitter` con:

- `chunk_size=500`
- `chunk_overlap=50`

Justificacion:

- Al probar segmentaciones por parrafo/punto con estrategias recursivas, aparecieron chunks muy cortos que reducian la continuidad semantica.
- Si un chunk queda demasiado pequeno, aumenta el riesgo de recuperar contexto incompleto.
- El overlap permite mantener continuidad entre fragmentos vecinos y mejora la probabilidad de recuperar informacion conectada.
- Un tamano de 500 busca equilibrio entre precision de recuperacion y cantidad de contexto enviado al modelo.

### Estrategia de embeddings y busqueda vectorial

- Se eligio `text-embedding-3-small` para mantener costos y latencia mas bajos, con buena calidad semantica para FAQs.
- Un embedding mas pequeno tambien reduce uso de memoria/almacenamiento en el vector store frente a alternativas mas grandes.
- La busqueda por similitud coseno es una opcion estandar para texto embebido, porque compara orientacion semantica entre vectores (mas que su magnitud).
- Para este caso (preguntas de lenguaje natural + FAQ), coseno suele funcionar bien para recuperar fragmentos conceptualmente cercanos.

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- API key de OpenAI

## Instalacion

### 1) Clonar el repositorio

```bash
git clone https://github.com/iv-chucoff/FAQs-HR-RAG.git
cd FAQs-HR-RAG
```

### 2) Instalar dependencias

Este comando crea el entorno virtual y sincroniza dependencias:

```bash
uv sync
```

### 3) Activar el entorno virtual

En Windows (PowerShell):

```powershell
.venv\Scripts\activate.ps1
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

### 4) Configurar variables de entorno

Crea un archivo `.env` en la raiz del proyecto:

```bash
OPENAI_API_KEY=tu-api-key-aqui
OPENAI_MODEL=gpt-4o-mini
```

Puedes obtener tu API key en [OpenAI Platform](https://platform.openai.com/api-keys).

## Uso

### Ejecucion basica

```bash
uv run python src/query.py --query "tu consulta aqui"
```

### Ejemplo

```bash
uv run python src/query.py --query "¿Como se hace una evaluacion 360?"
```

La salida es un JSON con:

- `user_question`
- `system_answer`
- `chunks_related`
- `evaluation` (`score` y `reason`)

Puedes ver ejemplos en `outputs/sample_queries.json`.

## Evaluacion de calidad de respuesta

El proyecto incluye una evaluacion automatica posterior a la respuesta RAG.

El evaluador puntua de 0 a 10 y justifica en texto:

- relevancia de chunks recuperados;
- calidad de la respuesta respecto al contexto;
- completitud de la respuesta;
- y penaliza contradicciones o invenciones.

Esto permite observar calidad de forma sistematica y detectar oportunidades de mejora en retrieval, prompt o chunking.

## Mejoras futuras

- Agregar metadata por categoria (ej. "Compensaciones", "Performance", etc.) para filtrar retrieval en Chroma.
- Ajustar estrategia de chunking por secciones semanticas del documento.
- Persistir y versionar mejor los indices para entornos de produccion.
- Crear un chatbot para dejar de pasar las queries por CLI.
