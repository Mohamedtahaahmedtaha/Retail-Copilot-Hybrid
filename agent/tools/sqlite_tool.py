import sqlite3
import os

class SQLiteTool:
    """
    A tool to execute read-only SQLite queries against the Northwind database.
    """

    def __init__(self, db_path="data/northwind.sqlite"):
        """
        Initializes the SQLite tool with the database path.
        """
        self.db_path = db_path

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at expected path: {db_path}. Please ensure it is present.")

    def get_schema_string(self):
        """
        Extracts the schema (table names and column definitions) as a string for the LLM.
        """
        conn = None
        schema_parts = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # الحصول على أسماء جميع الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                # الحصول على تعريفات الأعمدة لكل جدول
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [f"{col[1]} {col[2]}" for col in cursor.fetchall()] # col[1]=name, col[2]=type
                
                schema_parts.append(f"Table: {table_name}")
                schema_parts.append(f"Columns: ({', '.join(columns)})")
                schema_parts.append("-" * 20)
                
            return "\n".join(schema_parts)

        except Exception as e:
            return f"Error extracting schema: {e}"
        finally:
            if conn:
                conn.close()

    def run(self, sql_query):
        """
        Executes a read-only SQL query and returns the results as a list of dictionaries.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  
            cursor = conn.cursor()

            # Execute the query
            cursor.execute(sql_query)

            # Fetch all results
            results = cursor.fetchall()

            # Convert results to list of dicts for LLM readability
            output = [dict(row) for row in results]

            return {"result": output, "error": None} 

        except sqlite3.OperationalError as e:
            return {"result": [], "error": f"SQLite Error: {e}"}
        except Exception as e:
            return {"result": [], "error": f"General Database Error: {e}"} 
        finally:
            if conn:
                conn.close()

if __name__ == '__main__':
    tool = SQLiteTool()
    query = "SELECT COUNT(*) FROM Products;"
    result = tool.run(query) 
    print(result)
    schema = tool.get_schema_string()
    print("\n--- Database Schema ---")
    print(schema)