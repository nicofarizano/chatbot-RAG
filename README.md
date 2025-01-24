# Promtior AI Challenge

**Descripción:**  
Proyecto desarrollado como parte del desafío técnico de Promtior. Este chatbot utiliza LangChain, Ollama y Llama2 para responder preguntas basadas en documentos web y archivos PDF. El proyecto está configurado con un flujo de Despliegue Continuo (CD), automatizando el despliegue de cambios al entorno de producción en Azure App Services mediante GitHub Actions. La solución destaca por integrar recuperación de información y generación de respuestas precisas y contextuales. 

## Tecnologías utilizadas
- **LangChain**: Framework para la integración de LLMs.
- **Ollama**: Servidor de modelos Llama2.
- **FAISS**: VectorStore para recuperación de información.
- **FastAPI**: Framework para exponer la API del chatbot.
- **Uvicorn**: Servidor ASGI para ejecutar la aplicación.
- **GitHub Actions**: Herramienta para automatizar flujos de trabajo (CI/CD).
- **Azure App Services**: Plataforma en la nube utilizada para el despliegue continuo.

## Características principales
- Carga de datos desde múltiples fuentes: documentos web y PDFs.
- Recuperación de los fragmentos más relevantes usando FAISS.
- Resumen y generación de respuestas enfocadas en las preguntas.
- Implementación de una API REST para interactuar con el chatbot.