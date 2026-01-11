import os
from dotenv import load_dotenv, find_dotenv
from sshtunnel import SSHTunnelForwarder
import pymysql


def _safe_table_name(name: str) -> str:
    # 테이블명은 SQL 바인딩(%s) 불가 -> 문자열로 붙여야 하므로 최소 검증
    if not name or any(not (c.isalnum() or c == "") for c in name):
        raise ValueError(f"Invalid table name: {name}")
    return f"{name}"


def _open_conn(env_file=load_dotenv(find_dotenv(), override=True)):
    load_dotenv(find_dotenv(), override=True)

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    SSH_HOST = os.getenv("SSH_HOST")
    SSH_PORT = int(os.getenv("SSH_PORT", "22"))
    SSH_USER = os.getenv("SSH_USER")
    SSH_PRIVATE_KEY_PATH = os.getenv("SSH_PRIVATE_KEY_PATH")

    tunnel = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_pkey=SSH_PRIVATE_KEY_PATH,
        remote_bind_address=(DB_HOST, DB_PORT),
        local_bind_address=(DB_HOST, 0),
    )
    tunnel.start()

    conn = pymysql.connect(
        host=DB_HOST,
        port=tunnel.local_bind_port,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    return tunnel, conn


def _close(tunnel, conn):
    if conn:
        try:
            conn.close()
        except Exception:
            pass
    if tunnel:
        try:
            tunnel.stop()
        except Exception:
            pass


def check_db_connection(env_file: str = ".env"):
    tunnel = None
    conn = None
    try:
        tunnel, conn = _open_conn(env_file)

        # 아주 가벼운 쿼리로 연결 확인
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

        print("db 정상연결")

    finally:
        _close(tunnel, conn)


def fetch_table_data(table_name: str, env_file: str = ".env", limit=None, offset: int = 0):

    tunnel = None
    conn = None
    try:
        tunnel, conn = _open_conn(env_file)
        t = _safe_table_name(table_name)

        with conn.cursor() as cur:
            if limit is None:
                cur.execute(f"SELECT * FROM {t}")
            else:
                cur.execute(f"SELECT * FROM {t} LIMIT %s OFFSET %s",
                            (int(limit), int(offset)))
            return cur.fetchall()

    finally:
        _close(tunnel, conn)
