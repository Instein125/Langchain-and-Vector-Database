import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
import requests


dataset_path= 'hub://samman/langchain_course_jarvis_assistant'
print(dataset_path)

def get_documentation_urls():
    """Function to get documentation urls
    Return: List of urls"""
    return [
        '/docs/huggingface_hub/guides/overview',
        '/docs/huggingface_hub/guides/download',
        '/docs/huggingface_hub/guides/upload',
        '/docs/huggingface_hub/guides/hf_file_system',
        '/docs/huggingface_hub/guides/repository',
        '/docs/huggingface_hub/guides/search',
    ]

def construct_full_url(base_url, relative_url):
    """Function to construct full url combining the relatince and base url
    Args: base_url(string)
    relative_url(string)
    
    return:
    String"""
    return base_url+relative_url


def scrape_page_content(url):
    """Send get request to url and parse the html response
    Args: url(string)
    Return: string"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    text = soup.body.text.strip()
    # Remove non-ASCII characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    """
    loop through the list of relative urls and scrape content and append it the the content list
    Args: base_url(string)
    relative_urls(List)
    filename(string)
    
    Return: content(List)
    """
    content = []
    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scrapped_content = scrape_page_content(full_url)
        content.append(scrapped_content.rstrip('\n'))

    #Writing the scrapped content to a file
    with open(filename, 'w', encoding='utf-8') as f:
        for item in content:
            f.write("%s\n" %item)

    return content


def load_docs(root_dir, filename):
    """
    Load the documents from a file
    Args: root_dir(string)
    filename(string)
    
    Return: Docs(list)"""
    docs = []
    try:
        loader = TextLoader(file_path=os.path.join(root_dir, filename), encoding='utf-8')
        loaded_docs = loader.load_and_split()
        docs.extend(loaded_docs)
    
    except Exception as e:
        pass

    return docs

def split_docs(docs):
    """Fuction to split the documents into individuals chunks
    Args: Docs(List)
    Return:  chunks(List of individual chunks)"""

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, )
    chunks = text_splitter.split_documents(docs)
    return chunks


def main():
    base_url = 'https://huggingface.co'
    filename = 'content.txt'
    root_dir ='./'
    relative_urls = get_documentation_urls()

    content = scrape_all_content(base_url, relative_urls, filename)

    docs = load_docs(root_dir, filename)

    texts = split_docs(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
    db.add_documents(texts)


# Call the main function if this script is being run as the main program
if __name__ == '__main__':
    main()

