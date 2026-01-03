import os
import logging
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from rag_service import CorrectiveRAGService

def create_app() -> Flask:
    app = Flask(__name__)

    # Logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("rag_api")

    # Config
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data")).resolve()
    k = int(os.getenv("RAG_K", "4"))
    max_web_urls = int(os.getenv("MAX_WEB_URLS", "3"))

    # Init service once (startup)
    try:
        rag = CorrectiveRAGService(
            data_dir=data_dir,
            k=k,
            max_web_urls=max_web_urls,
        )
        log.info("RAG service initialized. DATA_DIR=%s", data_dir)
    except Exception as e:
        log.exception("Failed to initialize RAG service: %s", e)
        rag = None

    @app.get("/health")
    def health():
        ok = rag is not None
        return jsonify({"ok": ok, "data_dir": str(data_dir)}), (200 if ok else 503)

    @app.post("/rag")
    def rag_endpoint():
        if rag is None:
            return jsonify({"error": "Service not initialized. Check server logs."}), 503

        payload = request.get_json(silent=True) or {}
        q = payload.get("query", "")
        allow_web = bool(payload.get("allow_web", True))

        try:
            result = rag.ask(q, allow_web=allow_web)
        except ValueError as e:
            raise BadRequest(str(e))
        except Exception as e:
            logging.getLogger("rag_api").exception("RAG failure: %s", e)
            return jsonify({"error": "Internal error"}), 500

        return jsonify({
            "query": q,
            "answer": result.answer
        })

    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
