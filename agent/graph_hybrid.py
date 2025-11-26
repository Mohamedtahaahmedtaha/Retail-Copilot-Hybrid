from typing import TypedDict, Annotated, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
import operator
import json
import re
import sys
import os
import dspy
import requests  

class LocalOllama(dspy.LM):
    def __init__(self, model="tinyllama", base_url="http://127.0.0.1:11434"):
        super().__init__(model)
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url
        self.history = []

    def basic_request(self, prompt, **kwargs):
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                text = response.json().get('response', '')
                self.history.append({'prompt': prompt, 'response': text, 'kwargs': kwargs})
                return [text]
            else:
                print(f"Ollama API Error: {response.text}")
                return [""]
        except Exception as e:
            print(f"Connection Error to Ollama: {e}")
            return [""]

    def __call__(self, *args, **kwargs): 
        prompt = args[0] if args else ""
        return self.basic_request(prompt, **kwargs)


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)))) 

from .dspy_signatures import Router, Planner, NLtoSQL, Synthesizer, PlannerOutput, FinalAnswer
from .tools.sqlite_tool import SQLiteTool
from .rag.retrieval import RagRetriever

print("Code Reached Graph File!")

class AgentState(TypedDict):
    question: str
    format_hint: str
    rag_context: str
    retrieval_confidence: float
    planner_output: PlannerOutput
    sql_query: str
    sql_error: str
    final_answer_raw: str
    final_answer: FinalAnswer
    repair_count: Annotated[int, operator.add]
    citations: List[str]
    path: str
    
class PlaceholderLM(dspy.LM):
    def __init__(self): 
        self.name = "PLACEHOLDER"
        self.prompt_responses = {"default": "Error."}
    def __call__(self, prompt, **kwargs):
        return [self.prompt_responses["default"]]

try:
    print("Connecting to Ollama manually...")
    ollama_model = LocalOllama(model='phi3.5:3.8b-mini-instruct-q4_K_M')
    
    test = ollama_model("test")
    
    dspy.settings.configure(lm=ollama_model)
    print("DSPy connected to Ollama/tinyllama (via LocalOllama Class).")
    
except Exception as e:
    print(f"Connection failed: {e}. Using Placeholder.")
    dspy.settings.configure(lm=PlaceholderLM())

sql_tool = SQLiteTool()
retriever_tool = RagRetriever()
router_module = dspy.Predict(Router)
planner_module = dspy.ChainOfThought(Planner)
nltosql_module = dspy.ChainOfThought(NLtoSQL)
synthesizer_module = dspy.Predict(Synthesizer)


def route_question(state: AgentState):
    question = state["question"]
    if isinstance(dspy.settings.lm, PlaceholderLM):
        path = 'hybrid'
    else:
        raw_path = router_module(question=question).path
        
        if raw_path is None:
            path = 'hybrid'
        else:
            path = raw_path.lower().strip()
            
    if path not in ['rag', 'sql', 'hybrid']: path = 'hybrid'
    
    print(f"--- 1. ROUTER: Path chosen: '{path}' ---")
    return { "path": path, "repair_count": 0}

def retrieve_context(state: AgentState):
    question = state["question"]
    k = 5 if state["path"] in ['hybrid', 'rag'] else 3 
    
    results = retriever_tool.run(question, k=k)
    
    rag_context = ""
    chunk_ids = []
    
    for i, doc in enumerate(results):
        doc_id = doc.metadata.get("source", f"CHUNK-{i}")
        rag_context += f"Chunk ID: {doc_id}\nContent:\n{doc.text}\n---\n"
        chunk_ids.append(doc_id)
        
    print(f"--- 2. RETRIEVER: Retrieved {len(results)} chunks. ---")
    return {"rag_context": rag_context, "citations": chunk_ids}

def plan_and_extract(state: AgentState):
    question = state["question"]
    context = state["rag_context"]
    
    safe_default_output = PlannerOutput(context_chunks_used=[], date_range_start="all-time", 
                                        date_range_end="all-time", kpi_formula="N/A", entities=[])

    if isinstance(dspy.settings.lm, PlaceholderLM):
        return {"planner_output": safe_default_output}
        
    try:
        prediction = planner_module(question=question, context=context)
        
       
        raw_output = getattr(prediction, 'planner_output', None)
        
        if raw_output is None:
            raw_output = getattr(prediction, 'answer', None)
            
        if raw_output is None:
             raise ValueError("Planner output is None or missing 'planner_output'/'answer' attribute.")

        raw_output = raw_output.strip()
        
        planner_output = PlannerOutput.model_validate_json(raw_output)
        return {"planner_output": planner_output}
        
    except Exception as e:
        print(f"--- 3. PLANNER ERROR: Failed to parse output, returning safe defaults: {e} ---")
        return {"planner_output": safe_default_output}

def generate_sql(state: AgentState):
    question = state["question"]
    planner_output = state["planner_output"]
    format_hint = state["format_hint"]
    
    if planner_output is None:
        planner_output = PlannerOutput(context_chunks_used=[], date_range_start="all-time", 
                                       date_range_end="all-time", kpi_formula="N/A", entities=[])
    
    schema = sql_tool.get_schema_string()
    
    if isinstance(dspy.settings.lm, PlaceholderLM):
        sql_query = "-- Dummy SQL, LLM failed."
    else:
        constraints_str = (f"Date Range: {getattr(planner_output, 'date_range_start', 'all-time')} to {getattr(planner_output, 'date_range_end', 'all-time')} | "
                           f"KPI Formula: {getattr(planner_output, 'kpi_formula', 'N/A')} | Entities: {getattr(planner_output, 'entities', [])}")
                           
        prediction = nltosql_module(question=question, schema=schema, constraints=constraints_str, format_hint=format_hint)
        
        if prediction is None or not hasattr(prediction, 'sql_query'):
            sql_query = "-- Dummy SQL, LLM failed: No output from NLtoSQL."
        else:
            raw_sql = getattr(prediction, 'sql_query', None)
            
            if raw_sql is None:
                 sql_query = "-- Dummy SQL, LLM failed: SQL output was None."
            else:
                 sql_query = re.sub(r'```sql|```', '', raw_sql.strip()).strip()
            
        if state["sql_error"] and state["repair_count"] > 0:
              sql_query = f"-- REPAIR ATTEMPT {state['repair_count']} for previous error: {state['sql_error'][:50]}\n{sql_query}"
              
    print(f"--- 4. NL-SQL: Generated query: {sql_query[:50]}... ---")
    return {"sql_query": sql_query, "sql_error": None}

def execute_sql(state: AgentState):
    sql_query = state["sql_query"]
    
    if not sql_query or "SELECT" not in sql_query.upper() or isinstance(dspy.settings.lm, PlaceholderLM):
        err = "SQL query was empty or invalidly generated."
        return {"sql_error": err, "sql_result": {"columns": [], "rows": []}}
    
    result = sql_tool.run(sql_query) 
    
    if result["error"]:
        print(f"--- 5. EXECUTOR: SQL Error: {result['error']} ---")
        return {"sql_error": result["error"], "sql_result": {"columns": [], "rows": []}}
    else:
        print(f"--- 5. EXECUTOR: SQL Success. Returned {len(result['result'])} rows. ---")
        used_tables = re.findall(r'(?:FROM|JOIN)\s+("?[a-zA-Z_\\s]+"?)(?=\\s)', sql_query, re.IGNORECASE)
        citation_tables = set()
        for table in used_tables:
            if 'order details' in table.lower(): citation_tables.add('"Order Details"')
            elif 'orders' in table.lower(): citation_tables.add('Orders')
            elif 'products' in table.lower(): citation_tables.add('Products')
            elif 'customers' in table.lower(): citation_tables.add('Customers')
            elif 'categories' in table.lower(): citation_tables.add('Categories')
            elif 'suppliers' in table.lower(): citation_tables.add('Suppliers')
            
        current_citations = state.get("citations", [])
        return {"sql_result": {"columns": result["result"][0].keys() if result["result"] else [], "rows": result["result"]}, 
                "sql_error": None, "citations": list(set(current_citations) | citation_tables)}
def synthesize_answer(state: AgentState):
    question = state["question"]
    context = state["rag_context"]
    constraints = state["planner_output"]
    sql_result = state.get("sql_result", {"columns": [], "rows": []})
    format_hint = state["format_hint"]
    
    if hasattr(constraints, 'model_dump'):
        constraints_data = constraints.model_dump()
    elif hasattr(constraints, 'dict'):
        constraints_data = constraints.dict()
    else:
        constraints_data = {} 
        
    constraints_json_str = json.dumps(constraints_data)

    if isinstance(dspy.settings.lm, PlaceholderLM):
        return {"sql_error": "FAILED SYNTHESIS.", "final_answer_raw": "{}", "repair_count": 2}
    
    sql_result_str = f"Columns: {sql_result.get('columns', [])}\nRows: {sql_result.get('rows', [])[:10]}" 
    
    prediction = synthesizer_module(question=question, context=context, sql_result=sql_result_str, constraints=constraints_json_str, format_hint=format_hint) 
    
    raw_answer = getattr(prediction, 'final_answer', None)

    if raw_answer is None:
        print("--- 6. SYNTHESIZER ERROR: Prediction output is None. Attempting repair. ---")
        return {"sql_error": "Synthesizer returned no output (None).", 
                "final_answer_raw": "{}", 
                "repair_count": state["repair_count"] + 1}
    
    final_answer_json = re.sub(r'```json|```', '', raw_answer.strip()).strip()
    
    try:
        data = json.loads(final_answer_json)
        final_answer = FinalAnswer.model_validate(data)
        
        all_citations = set(state.get("citations", []))
        all_citations.update(final_answer.citations)
        final_answer.citations = list(all_citations)
        
        return {"final_answer": final_answer, "sql_error": None, "final_answer_raw": final_answer_json}
        
    except Exception as e:
        print(f"--- 6. SYNTHESIZER ERROR: Invalid final JSON output or structure: {e} ---")
        return {"sql_error": f"Invalid final output format/structure: {e}", 
                "final_answer_raw": final_answer_json, 
                "repair_count": state["repair_count"] + 1}
def get_next_path(state: AgentState):
    if state["sql_error"] and state["repair_count"] < 3: 
        print(f"--- 7. CHECK/REPAIR: Attempting repair #{state['repair_count'] + 1}. ---")
        return "REPAIR_SQL"
    else: 
        if state["sql_error"]:
            print("--- 7. TERMINATE (Fail): Max repair attempts reached. ---")
        return "TERMINATE"

workflow = StateGraph(AgentState)
workflow.add_node("router", route_question)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("plan", plan_and_extract)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("synthesize", synthesize_answer)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", lambda x: x["path"], {"rag": "retrieve", "sql": "generate_sql", "hybrid": "retrieve"})
workflow.add_edge("retrieve", "plan")
workflow.add_conditional_edges("plan", lambda x: "generate_sql" if x["path"] in ["hybrid", "sql"] else "synthesize", {"generate_sql": "generate_sql", "synthesize": "synthesize"})
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "synthesize")
workflow.add_conditional_edges("synthesize", get_next_path, {"REPAIR_SQL": "generate_sql", "TERMINATE": END})
app = workflow.compile()

def run_one(example):
    if os.path.join(os.getcwd(), 'agent') not in sys.path:
        sys.path.append(os.path.join(os.getcwd(), 'agent'))

    id = example.get("question_id")
    question = example.get("question")
    format_hint = example.get("format_hint")
    initial_state = AgentState(question=question, format_hint=format_hint, rag_context="", retrieval_confidence=0.0,
        planner_output=PlannerOutput(date_range_start="", date_range_end=""), sql_query="", sql_result={"columns": [], "rows": []},
        sql_error=None, final_answer=None, repair_count=0, citations=[], path="",)
    
    print(f"\n--- Running ID: {id} ---")
    
    if isinstance(dspy.settings.lm, PlaceholderLM):
        print("--- FATAL: Using PlaceholderLM. Aborting run. ---")
        return {"id": id, "final_answer": None, "sql": "", "confidence": 0.0, "explanation": "FATAL: Connection Failed", "citations": []}

    try:
        final_state = app.invoke(initial_state)
        final_answer_obj = final_state.get('final_answer')
        
        if final_answer_obj:
            return final_answer_obj.model_dump()
        else:
            error_explanation = final_state.get('sql_error') or "Agent failed."
            return {"id": id, "final_answer": None, "sql": final_state.get('sql_query', 'N/A'), "confidence": 0.0, "explanation": error_explanation, "citations": final_state.get('citations', [])}
            
    except Exception as e:
        print(f"FATAL ERROR for {id}: {e}")
        return {"id": id, "final_answer": None, "sql": initial_state.get('sql_query', 'N/A'), "confidence": 0.0, "explanation": f"FATAL SYSTEM ERROR: {e}", "citations": []}

def run_batch(batch_file, output_file):
    results = []
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading batch file {batch_file}: {e}")
        return

    print(f"Starting batch run of {len(questions)} questions...")
    for q in questions:
        result = run_one(q)
        result["id"] = q["question_id"]
        results.append(result)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    print(f"\n--- Batch processing complete. Results saved to {output_file} ---")