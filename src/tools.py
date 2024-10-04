from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from config import Config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()



class ChromaDB:
    def __init__(self, index_name):
        self.embedding_function = self.load_embedding_function()
        self.index_name = index_name
        self.documents = self.load_documents()
        self.embeddings = self.extract_embeddings()

    def load_embedding_function(self):
        return HuggingFaceBgeEmbeddings(
            model_name=Config().embedding_model_name,
            model_kwargs={'device': Config().device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def load_documents(self):
        urls = Config().data_urls
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        return docs_list

    def extract_embeddings(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Config().chunk_size, chunk_overlap=Config().chunk_overlap
        )
        docs_splits = text_splitter.split_documents(self.documents)
        vectorstore = Chroma.from_documents(
            documents=docs_splits,
            collection_name=self.index_name,
            embedding=self.embedding_function,
        )
        return vectorstore


class LLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config().llm_model_name,
            api_key=Config().api_key,
            base_url=Config().base_url,
            temperature=Config().temperature)


class WebSearchTool:
    def __init__(self):
        self.web_search_tool = self.load_web_tool()

    def load_web_tool(self):
        tavily_search = TavilySearchResults(k=Config().k,
                                     tavily_api_key=os.getenv("TAVILY_API_KEY"))
        web_search_tool = Tool(
            name="web_search",
            func=tavily_search.run,
            description=Config().web_search_description
        )
        return web_search_tool



class RetrieverTool:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever_tool = self.load_retriever_tool()

    def run_retriever(self, input_text):
        retriever = self.vectorstore.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": Config().k})
        return retriever.invoke(input_text, k=Config().k)

    def load_retriever_tool(self):
        retriever_tool = Tool(
            name="retriever",
            func=self.run_retriever,
            description=Config().retriever_description
        )
        return retriever_tool



class DocumentsGrader:
    def __init__(self, llm):
        self.llm = llm
        self.grade_prompt = self.load_grade_prompt()
        self.documents_llm_grader = self.load_documents_grader()

    def load_grade_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Config().grade_documents_prompt),
                ("human", "Retrieved document: \n\n {document} \n\n "
                          "User question: {question}"),
            ]
        )
        return prompt

    def load_documents_grader(self):

        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        retrieval_grader = self.grade_prompt | structured_llm_grader
        return retrieval_grader



class Generator:
    def __init__(self, llm):
        self.llm = llm
        self.rag_generator_prompt = Config().generate_prompt
        self.rag_generator = self.load_rag_generator()

    def load_rag_generator(self):
        rag_chain = self.rag_generator_prompt | self.llm | StrOutputParser()
        return rag_chain



class HallucinationsGrader:
    def __init__(self, llm):
        self.llm = llm
        self.hallucinations_prompt = self.load_hallucinations_prompt()
        self.hallucinations_grader = self.load_hallucinations_grader()

    def load_hallucinations_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Config().hallucination_prompt),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        return prompt

    def load_hallucinations_grader(self):

        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""
            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )

        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        hallucination_grader = self.hallucinations_prompt | structured_llm_grader
        return hallucination_grader



class AnswerGrader:
    def __init__(self, llm):
        self.llm = llm
        self.answer_grader_prompt = self.load_answer_prompt()
        self.answer_grader = self.load_answer_grader()

    def load_answer_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Config().answer_grader_prompt),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        return prompt

    def load_answer_grader(self):

        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""
            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no'"
            )

        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        answer_grader = self.answer_grader_prompt | structured_llm_grader
        return answer_grader



class QuestionRewriter:
    def __init__(self, llm):
        self.llm = llm
        self.rewrite_prompt = self.load_rewriter_prompt()
        self.question_rewriter = self.load_question_rewriter()

    def load_rewriter_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Config().question_rewriter_prompt),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        return prompt

    def load_question_rewriter(self):
        question_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()
        return question_rewriter