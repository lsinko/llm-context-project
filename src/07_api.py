import os
import sqlite3
from urllib.parse import quote

from flask import Flask, jsonify, request

DB_PATH = "data/processed/llm_context.db"

app = Flask(__name__)


def get_conn() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Nedostaje {DB_PATH}. Prvo pokreni src/06_store_db.py.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/models")
def models():
    provider = (request.args.get("provider") or "").strip()
    min_cw = request.args.get("min_context_window")
    max_cw = request.args.get("max_context_window")
    limit = request.args.get("limit", "200")

    try:
        limit_i = int(limit)
        if limit_i <= 0:
            limit_i = 200
        limit_i = min(limit_i, 2000)
    except ValueError:
        limit_i = 200

    where = []
    params = []

    if provider:
        where.append("provider = ?")
        params.append(provider)

    if min_cw is not None:
        try:
            where.append("context_window >= ?")
            params.append(int(min_cw))
        except ValueError:
            pass

    if max_cw is not None:
        try:
            where.append("context_window <= ?")
            params.append(int(max_cw))
        except ValueError:
            pass

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT
            kaggle_row_id, model_name, provider,
            context_window, latency_sec, speed_tokens_per_sec,
            benchmark_mmlu, benchmark_chatbot_arena,
            hf_repo_id, hf_status, hf_likes, hf_downloads, hf_downloads_all_time
        FROM llm_row
        {where_sql}
        ORDER BY context_window DESC
        LIMIT ?
    """
    params.append(limit_i)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return jsonify([dict(r) for r in rows])


@app.get("/repo/<path:hf_repo_id>")
def repo_detail(hf_repo_id: str):
    hf_repo_id = hf_repo_id.strip()

    with get_conn() as conn:
        repo = conn.execute("SELECT * FROM llm_repo WHERE hf_repo_id = ? LIMIT 1", (hf_repo_id,)).fetchone()
        if repo is None:
            return jsonify({"error": "Repo nije pronađen u bazi.", "hf_repo_id": hf_repo_id}), 404

       # prikazuje i sve retke s Kaggle koji sadrže navedeni repo
        rows = conn.execute(
            """
            SELECT
                kaggle_row_id, model_name, provider,
                context_window, latency_sec, speed_tokens_per_sec,
                benchmark_mmlu, benchmark_chatbot_arena
            FROM llm_row
            WHERE hf_repo_id = ?
            ORDER BY context_window DESC
            """,
            (hf_repo_id,),
        ).fetchall()

    out = dict(repo)
    out["kaggle_rows"] = [dict(r) for r in rows]
    return jsonify(out)


@app.get("/providers/summary")
def providers_summary():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                provider,
                COUNT(*) AS n_rows,
                AVG(context_window) AS avg_context_window,
                AVG(latency_sec) AS avg_latency_sec,
                AVG(speed_tokens_per_sec) AS avg_speed_tokens_per_sec,
                AVG(benchmark_mmlu) AS avg_benchmark_mmlu
            FROM llm_row
            GROUP BY provider
            ORDER BY n_rows DESC
            """
        ).fetchall()

    return jsonify([dict(r) for r in rows])


@app.get("/repos")
def repos():
    provider = (request.args.get("provider") or "").strip()
    limit = request.args.get("limit", "200")

    try:
        limit_i = int(limit)
        if limit_i <= 0:
            limit_i = 200
        limit_i = min(limit_i, 2000)
    except ValueError:
        limit_i = 200

    if provider:
        sql = """
            SELECT * FROM llm_repo
            WHERE provider = ?
            ORDER BY hf_downloads DESC
            LIMIT ?
        """
        params = (provider, limit_i)
    else:
        sql = """
            SELECT * FROM llm_repo
            ORDER BY hf_downloads DESC
            LIMIT ?
        """
        params = (limit_i,)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
