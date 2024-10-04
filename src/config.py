from langchain import hub



class Config():
    def __init__(self):
        self.chat_history_path = 'chat_history.txt'
        self.generation_limit = 3
        self.graduation_limit = 5
        self.data_urls = [
            "https://en.wikipedia.org/wiki/Computer_vision",
            "https://viso.ai/computer-vision/what-is-computer-vision/",
            "https://blog.theos.ai/articles/introduction-to-computer-vision#:~:text=Computer%20vision%20is%20composed%20of,and%20text%20to%20image%20generation",
            "https://www.geeksforgeeks.org/computer-vision-tasks/",
            "https://www.geeksforgeeks.org/introduction-convolution-neural-network/",
            "https://www.educative.io/answers/how-does-cnn-work-in-computer-vision-tasks"
        ]
        self.embedding_model_name = "BAAI/bge-m3"
        self.index_name = "rag-chroma"
        self.device = "cuda"
        self.k = 3
        self.chunk_size = 500
        self.chunk_overlap = 0

        self.llm_model_name = "llama3.1"
        self.api_key = 'ollama'
        self.base_url = 'http://localhost:11434/v1/'
        self.temperature = 0



        ### ----- Prompts -----

        # --- GradeDocuments ---
        self.grade_documents_prompt = """
        You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """

        # --- Generate ---
        self.generate_prompt = hub.pull("rlm/rag-prompt")


        #  --- HallucinationGrader ---
        self.hallucination_prompt = """
        You are a grader assessing whether an LLM generation is grounded in / supported by some of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts. 
     'no' means absolutely wrong generation
     """

        # --- AnswerGrader ---
        self.answer_grader_prompt = """
        You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. yes' means that the answer resolves the question.
     """

        # --- QuestionReWriter ---
        self.question_rewriter_prompt = """
        You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. 
     Give strictly only the improved question without any additional words and thoughts from you.
        """


        # ----- Tools -----
        self.retriever_description = ("Perform a retriever search to get information "
                                      "based on the query about Computer Vision(CV) "
                                      "and Convolution Neural Networks(CNN)")
        self.web_search_description = ("Perform a web search to look for any questions that is "
                                       "unrelated to Computer Vision(CV) "
                                       "and Convolution Neural Networks(CNN)")