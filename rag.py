from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from services.vector_store import VectorStore

load_dotenv()


vector_store = VectorStore()
retriever = vector_store._client.as_retriever()

template = """Answer the question based only on the following context:
{content}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"content": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
