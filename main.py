import os
os.environ["USER_AGENT"] = "promtior-bot"

import argparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL = "llama2"  # Modelo que usará Ollama

def summarize_documents(llm, docs, question, max_words=200): # Resume los documentos recuperados utilizando el modelo de lenguaje, enfocado en la pregunta inicial.
    combined_text = " ".join([doc.page_content for doc in docs])
    summary_prompt = (
        f"A continuación, tienes un texto. Responde a la siguiente pregunta basándote en el texto. "
        f"Limita tu respuesta a no más de {max_words} palabras.\n\n"
        f"Pregunta: {question}\n\n"
        f"Texto:\n{combined_text}"
    )
    summary = llm.invoke(summary_prompt)
    return summary

def main():
    # Configurar argumentos para la URL y el PDF
    parser = argparse.ArgumentParser(description="Cargar datos desde la web y un PDF para preguntas con Llama 2.")
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.promtior.ai",
        required=False,
        help="La URL para cargar datos."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="AI Engineer.pdf",
        required=False,
        help="Ruta al archivo PDF para cargar."
    )

    args = parser.parse_args()
    url = args.url
    pdf_path = args.pdf

    print(f"Usando la URL: {url}")
    print(f"Usando el archivo PDF: {pdf_path}")

    # Cargar datos desde la web
    web_loader = WebBaseLoader(url)
    web_data = web_loader.load()

    # Cargar datos desde el PDF
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_data = pdf_loader.load()

    # Dividir datos en fragmentos manejables
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    web_splits = text_splitter.split_documents(web_data)
    pdf_splits = text_splitter.split_documents(pdf_data)

    # Combinar todos los fragmentos
    all_splits = web_splits + pdf_splits
    print(f"Total de fragmentos: {len(all_splits)}")

    # Crear el VectorStore con FAISS y GPT4AllEmbeddings
    embedding_model = GPT4AllEmbeddings()
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
    print("VectorStore creado con éxito.")

    # Configurar el modelo Ollama con Llama 2
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = OllamaLLM(model=MODEL, callback_manager=callback_manager)
    print(f"Modelo {MODEL} cargado.")

    # Interacción en tiempo real con el usuario
    while True:
        question = input("\nEscribe tu pregunta (o escribe 'salir' para terminar): ").strip()
        if question.lower() == "salir":
            print("Saliendo de la prueba técnica. ¡Adiós!")
            break
        
        # Recuperar los documentos más relevantes
        docs = vectorstore.similarity_search(question, k=3)  # Recuperar 3 documentos
        print(f"\nDocumentos recuperados: {len(docs)}")
        
        # Resumir los documentos recuperados
        print("\nRespuesta generada: ")
        summarize_documents(llm, docs, question, max_words=200)

if __name__ == "__main__":
    main()