import os
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

def load_database(dataset_path):
    """Load the deeplake datastore"""
    embeddings =  GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    db = DeepLake(dataset_path, read_only=True, embedding=embeddings)
    return db



def main():
    dataset_path = "hub://samman/twitter-algorithm"
    db = load_database(dataset_path)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10

    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    #Ask Questions to the Codebase in Natural Language
    questions = [
        "What does favCountParams do?",
        "is it Likes + Bookmarks, or not clear from the code?",
        "What are the major negative modifiers that lower your linear ranking parameters?",   
        # "How do you get assigned to SimClusters?",
        # "What is needed to migrate from one SimClusters to another SimClusters?",
        # "How much do I get boosted within my cluster?",   
    ] 
    chat_history = []

    for question in questions:  
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")



if __name__ == '__main__':
    main()


