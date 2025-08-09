from flask import Flask, render_template, jsonify, request
#from langchain.chains import create_retrival_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from src.prompt import *
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore

#from langchain.llms import CTransformers
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY


embeddings = download_hugging_face_embedding()

#Initializing the Pinecone
"""Pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_KEY)"""

index_name="medicalbot"

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
#PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#chain_type_kwargs={"prompt": PROMPT}

llm = OllamaLLM(
    model="llama3",
    temperature=0.8,
    num_ctx=512,                 # raise if you need bigger context
    base_url="http://localhost:11434"
)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system_prompt),
    ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

"""qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)"""



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    #result=qa({"query": input})
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)