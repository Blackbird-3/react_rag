from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['GET'])
def server_status():
    return jsonify({"status": "server running"})

@app.route("/answer" , methods = ['POST'])
def answer():
    os.environ["PINECONE_API_KEY"] = "pcsk_3JC6xT_2xm7TneUZK8EiVtVbRmUkuaZDTNJdswxNLQEZkhNViZXoU79T5JHwgKzS4fmkKn"
    os.environ["GROQ_API_KEY"]= "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"
    data= request.json
    index_name = "rag-fullstack"
    query = data.get('query')
    print(query)
    llm=ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.5
    )
    knowledge = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=OllamaEmbeddings(model="llama3.1"))
    qa= RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever()
    )
    res = "I am Empty"
    try:
        res = qa.invoke(query).get("result")
    except Exception as error:
        res = "error in getting data : "+error
    return jsonify({"answer": res})
    # return jsonify({"answer": qa.invoke(query).get("result")})

if __name__ == "__main__":
    # app.run(debug=True , port = 10000)   
    port = int(os.environ.get("PORT", 11434 )) 
    app.run(debug=True,host='0.0.0.0', port=port)
