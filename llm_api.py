import warnings
warnings.filterwarnings("ignore")

import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import unstructured
import sys
import torch
from torch.utils._pytree import register_pytree_node
import ipywidgets as widgets
from IPython.display import display
import os
import subprocess
import pyterrier as pt
import pandas as pd
from transformers import pipeline

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from typing import Dict, Any

import time
import yaml

dict_local_parameters = yaml.safe_load(open(sys.argv[1], "r"))
dict_global_parameters = yaml.safe_load(open(dict_local_parameters["global parameters path"], "r"))

app = FastAPI()

print("pre init")
pt.init()
print("post init")

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

chat_history = []

# Specify the path to the saved model
new_model_path = "./Llama-2-7b-chat-hf"

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(new_model_path,torch_dtype=torch.float16,
                                             #use_auth_token=True,
                                             load_in_8bit=True,)
print(next(model.parameters()).is_cuda)
#model.to('cuda')
#print(next(model.parameters()).is_cuda)

time.sleep(10)
tokenizer = AutoTokenizer.from_pretrained(new_model_path)

                                             
print("Model created!")

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 1024,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

text_data = []
Doc_Names =  ["Fengyun-2D", "Fengyun-2E", "Fengyun-2F", "Fengyun-2H", "Fengyun-4A",
    "Sentinel-3A", "Sentinel-3B", "Sentinel-6A", "Jason-1", "Jason-2", "Jason-3",
    "CryoSat-2", "Haiyang-2A", "TOPEX", "Summary", "SARAL"
]

i = 0
for sat in Doc_Names:
    file_name = f'{dict_global_parameters["llm files directory"]}{sat}.txt'
    with open(file_name, 'r') as file:
      file_text = file.read()
      text_data.append({'docno': i, 'title':sat, 'text': file_text})
    i = i+1

df1 = pd.DataFrame(text_data)




directory_path = './pd_index'

subprocess.run(['rm', '-rf', directory_path])
#pd_indexer = pt.DFIndexer("./pd_index")


df1["text"] = df1["text"].astype(str)
df1["title"] = df1["title"].astype(str)
df1["docno"] = df1["docno"].astype(str)




pd_indexer = pt.DFIndexer(index_path="./pd_index")


indexref = pd_indexer.index(df1["title"], df1["text"], df1["docno"])

index = pt.IndexFactory.of(indexref)

index = pt.IndexFactory.of('./pd_index/data.properties')

pd_indexer.setProperty("tokeniser", "UTFTokeniser")
pd_indexer.setProperty( "termpipelines", "Stopwords,PorterStemmer")

data1 = []
for row in df1.itertuples():
    data1.append({'docno':row.docno, 'title': row.title, 'body': row.text})
iter_indexer = pt.IterDictIndexer("./pd_index", meta={'docno': 20, 'title': 10000, 'body':100000},
overwrite=True, blocks=True)
RETRIEVAL_FIELDS = ['title','body']
indexref1 = iter_indexer.index(data1, fields=RETRIEVAL_FIELDS)


import warnings

warnings.filterwarnings("ignore", message="Created a chunk of size .*")



def QAbot(query, chat_history):

    
	query = query.replace('?','')
	print(f'QA query:{query}')
	print(f'QA Query type:{type(query)}')


	qe = (pt.rewrite.SequentialDependence() >> 
    	pt.BatchRetrieve(indexref1, wmodel="BM25"))

	
	
	result = qe.search(query)
	
    
	print("terrier result", result)
	title = df1["title"][int(result["docno"][0])]
	print(f'Document chosen: {title}')
	
	loader = UnstructuredFileLoader(dict_global_parameters["llm files directory"]+str(title)+".txt")
	documents = loader.load()

	#print("Document loader is done!")

	text_splitter=CharacterTextSplitter(separator='\n',
                                    chunk_size=1500,
                                    chunk_overlap=50)

	text_chunks=text_splitter.split_documents(documents)

	embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})

	llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.5})
	vectorstore = FAISS.from_documents(text_chunks, embeddings)
	#chain =  ConversationalRetrievalChain.from_llm(llm=llm, chain_type = "stuff",return_source_documents=True, retriever=vectorstore.as_retriever(), get_chat_history = None)
	template = (
		"Combine the chat history and follow up question into "
		"a standalone question. Chat History: {chat_history}"
		"Follow up question: {question}"
		"""Use the following pieces of information to answer the user's question.
		If you don't know the answer, just say that you don't know, don't try to make up an answer.
		Context: {context}
		Question: {question}
		Only return the helpful answer below and nothing else.
		Helpful answer:
		""")
	prompt = PromptTemplate.from_template(template)
	chain =  ConversationalRetrievalChain.from_llm(
		llm=llm,
		prompt=prompt,
		chain_type = "stuff",
		return_source_documents=False,
		return_generated_question = True,
		retriever=vectorstore.as_retriever(),
	)
	#print(chat_history)
	#chain =  RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff",return_source_documents=True, retriever=vectorstore.as_retriever())
	#result=chain.invoke({"question": query, "chat_history": chat_history}, return_only_outputs=True)
	result = chain({"question": query, "chat_history": chat_history})
	print("the result:", result)
	return result['answer']
  
print("QAbot created!")
  
@app.post("/chatbot_query/")
async def chatbot_query(request: Dict[Any, Any]):
	chat_history=[]
	print("request received")
	query = request["query"]
	print(f'query:{query}')
	print(f'Query type:{type(query)}')
	answer = QAbot(query, chat_history)
	print(f'answer:{answer}')
	return {"answer": answer}

# @app.post("/input_query/")
# async def input_query(request: Request):

#         chat_history=[]

#         data = await request.json()

#         print(data["query"])
#         print(f'Query type: {type(data["query"])}')

#         answer = QAbot(data["query"], chat_history)

#         return {"answer": answer}
		

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2222)
