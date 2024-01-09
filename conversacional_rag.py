from operator import itemgetter

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearchVectorStoreRetriever
from langchain_core.messages import HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from services.vector_store import VectorStore

load_dotenv()


vector_store = VectorStore()
retriever = AzureSearchVectorStoreRetriever(
    vectorstore=vector_store._client, search_type="semantic_hybrid"
)


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

system_template = (
    "You are an expert winemaker. You have vast knowledge about wines. You "
    "are friendly, helpful, and funny. You keep your answers short, usually "
    "just a paragraph. Keep your answers mostly based on the context. You can "
    "do things like recommend wines mentioned in the context, describe wines"
    "mentioned in the context, talk about wineries, varieties, regions, "
    "locations and tastes mentioned in the context. You are allowed to add "
    "colored facts if they add information to the question beign answered."
)
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages([system_prompt, template])

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{name}: [{category}][{subcategory}] {page_content}"
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

if __name__ == "__main__":
    chat_history = []
    while True:
        human_message = input()
        ai_message = conversational_qa_chain.invoke(
            {"question": human_message, "chat_history": chat_history}
        )
        chat_history.append(HumanMessage(content=human_message))
        chat_history.append(ai_message)
        print(get_buffer_string(chat_history))
