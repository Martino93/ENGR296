import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract

load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.document_loaders import ReadTheDocsLoader, PyPDFLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")
PDF_DIR = "./documents"


def ingest_documents(image=False, image_path=None):
    """
    This function is used to ingest documents into Pinecone.

    args:
    image: bool, default=False
        If True, the function will handle scanned PDFs
    """

    if not image:
        loader = PyPDFDirectoryLoader(PDF_DIR)
        documents = loader.load()

        print(f"Loaded {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=30)
        text_splitter.split_documents(documents)

        for doc in documents:
            location = doc.metadata.get("source")
            print(location)

        print(f"Adding {len(documents)} documents to Pinecone")

        PineconeVectorStore.from_documents(
            documents, EMBEDDINGS, index_name=os.environ["LANGCHAIN_PROJECT"]
        )

    else:
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Users\Hamed\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        )
        pages = convert_from_path(image_path, dpi=500)

        full_text = []
        for page_number, page in enumerate(pages, start=1):
            text = pytesseract.image_to_string(page)
            full_text.append(f"Page {page_number}\n{text}")

        extracted_text = "\n".join(full_text)

        print(extracted_text)
        # # split text into chunks
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=30)

        # text_splitter.split_text(extracted_text)

        # PineconeVectorStore.from_texts(
        #     extracted_text, EMBEDDINGS, index_name=os.environ["LANGCHAIN_PROJECT"]
        # )


if __name__ == "__main__":
    # file_to_load = './documents/labs_and_vitals.pdf'
    ingest_documents()
