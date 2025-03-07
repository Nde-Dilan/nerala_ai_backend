import tika
"""
PDF Injection Processing Module
This module contains utilities for extracting text from PDFs, splitting the text
into chunks, generating embeddings, and storing them in a vector database.
Imports:
    - tika: A toolkit for extracting text from various file formats
    - tika.parser: Provides functionality to parse documents and extract content
    - langchain.text_splitter.CharacterTextSplitter: Splits text into manageable 
      chunks based on character count
    - langchain.embeddings.HuggingFaceEmbeddings: Creates vector embeddings of text
      using HuggingFace models
    - langchain.vectorstores.Chroma: Vector database for storing and retrieving
      embeddings for semantic search
"""
from tika import parser
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


text = parser.from_file("./Dictionnaire-Fulfulde-français-english-et-images.pdf")
