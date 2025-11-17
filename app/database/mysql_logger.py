import mysql.connector
import os
from dotenv import load_dotenv
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASS", "")
MYSQL_DB = os.getenv("MYSQL_DB", "tcs_forecasts")

def get_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        autocommit=True
    )

def log_request(request_id: str, input_query: str, response_json: str):
    conn = get_conn()
    cursor = conn.cursor()
    sql = "INSERT INTO request_logs (request_id, input_query, response_json) VALUES (%s, %s, %s)"
    cursor.execute(sql, (request_id, input_query, response_json))
    cursor.close()
    conn.close()

def save_document(source, url, filename, content):
    conn = get_conn()
    cursor = conn.cursor()
    sql = "INSERT INTO raw_documents (source, url, filename, content) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (source, url, filename, content))
    cursor.close()
    conn.close()
