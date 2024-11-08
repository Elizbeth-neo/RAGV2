import re
import json
from config import Config
from langgraph.prebuilt import tools_condition
from langchain.schema import Document
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()



class RAGGraph:
    def __init__(self, llm, tools, retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_rewriter):
        self.llm = llm
        self.tools = tools
        self.retrieval_grader = retrieval_grader
        self.rag_chain = rag_chain
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.question_rewriter = question_rewriter
        self.workflow = self.build_workflow()


    def run_chatbot(self, inputs):
        app = self.workflow.compile()
        value = app.invoke(inputs)
        return value

    def build_workflow(self):

        class ToolCall(TypedDict):
            id: str
            function: dict
            type: str


        class GraphState(TypedDict):
            """
            Represents the state of the graph.
            """
            question: str
            generation: str
            generation_id: int
            docs_grade_id: int
            tool_cals: List[ToolCall]
            tool_name: str
            messages: List[str]
            chat_history: List[str]
            documents: List[str]


        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.
            """

            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            messages = state['messages']
            tool_name = state['tool_name']
            docs_grade_id = state["docs_grade_id"]
            docs_grade_id +=1
            print(f'---> grade_documents iteration: {docs_grade_id}')

            if tool_name =='retriever':
                docs = messages[-1].content
                doc_strings = re.findall(r'Document\(metadata=(.+?), page_content=(.+?)\)', docs)
                documents = []
                for metadata_str, page_content_str in doc_strings:
                    metadata = eval(metadata_str)
                    page_content = page_content_str.strip("'")
                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)
            else:
                tool_message_content = messages[-1].content
                try:
                    parsed_content = json.loads(tool_message_content)
                    documents = [Document(page_content=item["content"], metadata={"source": item["url"]}) for item in parsed_content]
                except Exception as e:
                    print(e)


            filtered_docs = []
            for d in documents:
                try:
                    score = self.retrieval_grader.invoke(
                        {"question": question, "document": d.page_content}
                    )
                    grade = score.binary_score
                except Exception as e:
                    return(e)
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    continue
            return {"documents": filtered_docs, "question": question, "docs_grade_id": docs_grade_id}


        def generate(state):
            """
            Generate answer
            """
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]
            chat_history = state["chat_history"]
            generation_id = state["generation_id"]
            generation_id += 1

            # RAG generation
            try:
                generation = self.rag_chain.invoke({"context": documents, "question": question})
            except Exception as e:
                return(e)

            chat_history.append([{
                "generation_id": generation_id,
                "user_question": question,
                "ai_answer": generation
            }])
            return {"documents": documents, "question": question,
                    "generation": generation, "generation_id": generation_id,
                    "chat_history": chat_history}


        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.
            """

            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]
            generation_id = state["generation_id"]
            tool_name = state["tool_name"]

            if tool_name == "retriever":
                docs_splitted = documents
            else:
                docs_splitted = [d.page_content for d in documents]
            try:
                score = self.hallucination_grader.invoke(
                    {"documents": docs_splitted, "generation": generation}
                )
            except Exception as e:
                return(e)

            # Check hallucinations
            if score:
                grade = score.binary_score
                if grade == "yes":
                    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                    # Check question-answering
                    print("---GRADE GENERATION vs QUESTION---")
                    try:
                        score = self.answer_grader.invoke({"question": question, "generation": generation})
                    except Exception as e:
                        return(e)
                    if score:
                        grade = score.binary_score
                        if grade == "yes":
                            print("---DECISION: GENERATION ADDRESSES QUESTION---")
                            return "useful"
                        else:
                            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                            return "not useful"
                    # To avoid a possible endless loop
                    print(f"~~~~~Failed answer grader~~~~~")
                    return "failed"
                else:
                    if generation_id <= Config().generation_limit:
                        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                        return "not supported"
                    else:
                        # To avoid a possible endless loop
                        print(f"~~~~~The generation limit has been reached~~~~~")
                        return "failed"
            else:
                # To avoid a possible endless loop
                print(f"~~~~~Failed hallucination grader~~~~~")
                return "failed"


        def decide_to_generate(state):
            """
            Determines whether to generate an answer, or re-generate a question.
            """
            print("---ASSESS GRADED DOCUMENTS---")
            filtered_documents = state["documents"]
            docs_grade_id = state["docs_grade_id"]

            if not filtered_documents:
                if docs_grade_id < Config().graduation_limit:
                    # No valid docs --> re-generate a new query
                    print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
                    return "transform_query"
                else:
                    # To avoid a possible endless loop
                    print(f"~~~~~The generation limit has been reached~~~~~")
                    return 'limit'
            else:
                # Relevant documents --> generate answer
                print("---DECISION: GENERATE---")
                return "generate"


        def transform_query(state):
            """
            Transform the query to produce a better question.
            """
            print("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]

            # Re-write question
            try:
                better_question = self.question_rewriter.invoke({"question": question})
            except Exception as e:
                return(e)
            print(f'-question-: {question}')
            print(f'\n')
            print(f'-transformed question-: {better_question}')
            return {"documents": documents, "question": better_question}


        def agent(state):
            """
            Invokes the agent model to generate a response based on the current state. Given
            the question, it will decide to retrieve using the retriever tool, or simply end.
            """
            print("---CALL AGENT---")
            question = state["question"]
            messages = state.get("messages", [])
            generation_id = state.get("generation_id", 0)
            docs_grade_id = state.get("docs_grade_id", 0)

            model = self.llm.bind_tools(self.tools)
            try:
                response = model.invoke(question)
            except Exception as e:
                return(e)
            messages.append(response)

            tool_name = response.additional_kwargs.get('tool_calls')[0]['function']['name']
            print(f'called ----> TOOL NAME ---> {tool_name}')
            return {
                "generation": response,
                "messages": messages,
                "tool_name": tool_name,
                "tool_calls": state.get("tool_calls", []),
                "question": state["question"],
                "generation_id": generation_id,
                "docs_grade_id": docs_grade_id,
                "documents": state.get("documents", []),
                "chat_history": state.get("chat_history", [])
            }


        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("agent", agent)
        tool_node = ToolNode(self.tools)
        workflow.add_node("action", tool_node)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)

        # Build graph
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "action"
            },
        )
        workflow.add_edge("action", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "limit": END
            },
        )
        workflow.add_edge("transform_query", "agent")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "failed": END,
                "limit": END,
                "useful": END,
                "not useful": "transform_query",
            },
        )
        return workflow




























