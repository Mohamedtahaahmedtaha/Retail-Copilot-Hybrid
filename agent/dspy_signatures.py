import dspy
from dspy import Signature

class Router(Signature):
    """
    Decide the appropriate tool/path to answer the user's question.
    Options: 'sql', 'rag', or 'hybrid'.
    - 'sql': For questions that require only structured data analysis (e.g., counts, totals, comparisons of quantitative data).
    - 'rag': For questions that require only document retrieval (e.g., policy details, definitions, marketing plans).
    - 'hybrid': For questions that require both SQL and document data (e.g., 'What is the return policy for the top-selling product category?').
    """
    question: str = dspy.InputField(desc="The user's question.")
    path: str = dspy.OutputField(desc="The chosen path: 'sql', 'rag', or 'hybrid'.")

class Planner(Signature):
    """
    Analyze the user's question to determine the necessary steps to answer it, especially focusing on how structured (SQL) data and unstructured (RAG) data might be combined.
    """
    question: str = dspy.InputField(desc="The user's question, which requires SQL or Hybrid path.")
    plan: str = dspy.OutputField(desc="A step-by-step plan (1-2 steps) to solve the question. Example: 1. Identify the top category via SQL. 2. Retrieve policy for that category.")

class NLtoSQL(Signature):
    """
    Convert a natural language question into an executable SQLite SQL query for the Northwind database. 
    The query must be syntactically correct and target the necessary tables (Customers, Orders, Products, Categories, Suppliers, etc.) to calculate the final quantitative answer.
    """
    question: str = dspy.InputField(desc="The user's question, requiring a database query.")
    sql_query: str = dspy.OutputField(desc="The final, executable SQLite query.")

class Synthesizer(Signature):
    """
    Synthesize the final, human-readable answer based on the provided user question and all available context (SQL results, retrieved documents). 
    Ensure the answer directly addresses the question using the context.
    """
    question: str = dspy.InputField(desc="The user's original question.")
    context: str = dspy.InputField(desc="All necessary information (SQL results, document chunks) to formulate the answer.")
    answer: str = dspy.OutputField(desc="The final, comprehensive answer to the user's question.")

class PlannerOutput(dspy.Prediction):
    plan: str
    sql_query: str

class FinalAnswer(dspy.Prediction):
    answer: str