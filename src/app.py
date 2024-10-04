import os
import tools
import graph_utils
from pprint import pprint
from loguru import logger as logging
from config import Config

from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

class ChatBot:

    def __init__(self):
        self.model = self.load_model()
        self.tools = self.load_tools()
        self.retrieval_grader = self.load_retrieval_grader()
        self.rag_chain = self.load_rag_chain()
        self.hallucination_grader = self.load_hallucination_grader()
        self.answer_grader = self.load_answer_grader()
        self.question_rewriter = self.load_question_rewriter()
        self.workflow = self.build_graph()

    def build_graph(self):
        workflow = graph_utils.RAGGraph(llm=self.model,
                                        tools=self.tools,
                                        retrieval_grader=self.retrieval_grader,
                                        rag_chain=self.rag_chain,
                                        hallucination_grader=self.hallucination_grader,
                                        answer_grader=self.answer_grader,
                                        question_rewriter=self.question_rewriter
                                        )
        return workflow

    def load_retrieval_grader(self):
        return tools.DocumentsGrader(self.model).documents_llm_grader

    def load_rag_chain(self):
        return tools.Generator(self.model).rag_generator

    def load_hallucination_grader(self):
        return tools.HallucinationsGrader(self.model).hallucinations_grader

    def load_answer_grader(self):
        return tools.AnswerGrader(self.model).answer_grader

    def load_question_rewriter(self):
        return tools.QuestionRewriter(self.model).question_rewriter

    def load_tools(self):
        vectorstore = tools.ChromaDB(Config().index_name).embeddings
        retriever_tool = tools.RetrieverTool(vectorstore=vectorstore).retriever_tool
        web_search_tool = tools.WebSearchTool().web_search_tool
        return [retriever_tool, web_search_tool]

    def load_model(self):
        return tools.LLM().llm



def main():
    chatbot = ChatBot()
    logging.info("ChatBot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            inputs = {
                "question": user_input,
            }
            answer = chatbot.workflow.run_chatbot(inputs)
            if len(answer["documents"])!=0:
                pprint(f"Answer: {answer['generation']}")
            else:
                pprint(answer)


    history = answer["chat_history"]
    history_path = Config().chat_history_path
    with open(history_path, "w") as text_file:
        text_file.write(str(history))
    logging.info(f'Chat history saved at: {history_path}')



if __name__ == "__main__":
    main()