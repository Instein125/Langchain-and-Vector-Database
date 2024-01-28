import os
from langchain.vectorstores import DeepLake
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def index_algorithm(root_dir):
    """Index every files in the-algorithm folder using TextLoader
    """
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
                # print(docs)
            except Exception as e:
                print(e)

    return docs

def create_chunks(docs):
    """Convert the loaded files into chunks"""
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(documents=docs)
    return texts

def store_embeddings(texts):
    """store the embedding into the vector database"""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    username = 'samman'
    db = DeepLake(dataset_path=f"hub://{username}/twitter-algorithm", embedding=embeddings)
    db.add_documents(texts)


def main():
    root_dir ='projects/understanding twitter algoritm/the-algorithm'
    docs = index_algorithm(root_dir=root_dir)

    texts = create_chunks(docs)

    store_embeddings(texts)


if __name__ == '__main__':
    main()
