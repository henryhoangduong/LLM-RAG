from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from retriever import retriever
load_dotenv()


class MultiQueryRetrieval:
    """
    This class retrieves documents using multi-query perspectives and
    generates an answer based on the retrieved documents.

    Parameters
    ----------
    retriever:
        The retriever object that will be used to retrieve documents.

    temperature: float
        The temperature parameter for the language model.

    Methods
    -------
    retrieve_documents:
        Retrieve documents using multi-query perspectives.

    answer_question:
        Generate an answer to the question based on retrieved documents.

    run:
        Run the multi-query retrieval and answering process.
    """

    def __init__(self, retriever, temperature=0):
        if not retriever:
            raise ValueError("Retriever cannot be None")

        self.retriever = retriever
        self.temperature = temperature
        self.method_name = "multi_query"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Initialize prompt for generating query perspectives
        self.prompt_perspectives = ChatPromptTemplate.from_template(
            """You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines. Original question: {question}"""
        )

        self.generate_queries = (
            self.prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        self.rag_template = ChatPromptTemplate.from_template(
            """Answer the following question based on this context:
            {context}

            Question: {question}
            """
        )

    @staticmethod
    def get_unique_union(documents: list[list]):
        """
        Unique union of retrieved docs from multiple queries.

        Parameters
        ----------
        documents: list[list]
            List of lists of documents.

        Returns
        -------
        list
            Unique union of documents.

        Raises
        ------
        ValueError
            If documents list is empty.
        """

        if not documents:
            raise ValueError("Documents list cannot be empty")

        try:
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]
        except Exception as e:
            raise ValueError(f"Failed to process documents: {str(e)}")

    def retrieve_documents(self, question: str):
        """
        Retrieve documents using multi-query perspectives

        parameters
        ----------
        question: str
            The question.

        returns
        -------
        list
            A list of retrieved documents.

        raises
        ------
        ValueError
            If the question is empty or no documents are retrieved.
        """
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            retrieval_chain = (
                self.generate_queries | self.retriever.map() | self.get_unique_union
            )
            docs = retrieval_chain.invoke({"question": question})
            if not docs:
                raise ValueError("No documents retrieved for the given question.")
            return docs
        except Exception as e:
            raise ValueError(f"Failed to retrieve documents: {str(e)}")

    def answer_question(self, question: str, docs: list):
        """
        Generate an answer to the question based on retrieved documents

        parameters
        ----------
        question: str
            The question.

        docs: list
            A list of retrieved documents.

        returns
        -------
        str
            The answer.

        raises
        ------
        ValueError
            If no documents are retrieved to generate an answer.
        """
        if not docs:
            raise ValueError("No documents retrieved to generate an answer")

        try:
            context = "\n".join([doc.page_content for doc in docs])
            print(context)
            final_rag_chain = (
                {"context": lambda x: context, "question":lambda x: question}
                | self.rag_template
                | self.llm
                | StrOutputParser()
            )
            return final_rag_chain.invoke({"question": question})
        except Exception as e:
            raise ValueError(f"Failed to generate answer: {str(e)}")

    def run(self):
        """
        Run the multi-query retrieval and answering process

        parameters
        ----------
        None

        implements
        ----------
        retrieve_documents:
            Retrieve documents using multi-query perspectives.

        answer_question:
            Generate an answer to the question based on retrieved documents.
        """
        question = input("Please enter your question: ")
        try:
            docs = self.retrieve_documents(question)
            answer = self.answer_question(question, docs)
            print("Answer:", answer)
        except ValueError as e:
            print(f"Error: {e}")

if __name__ =="__main__":
    multiQueryRetrieval = MultiQueryRetrieval(retriever=retriever)
    print(multiQueryRetrieval.run())
