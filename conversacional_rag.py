from operator import itemgetter

from dotenv import load_dotenv
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from services.vector_store import VectorStore

load_dotenv()


vector_store = VectorStore()
retriever = vector_store._client.as_retriever()


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

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
        for message in chat_history:
            if type(message) == HumanMessage:
                print("You: ", end="")
            else:
                print("Assistant: ", end="")
            print(message.content)
