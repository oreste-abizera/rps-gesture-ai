from flask import jsonify, Blueprint, current_app

try:
    from flask_swagger_ui import get_swaggerui_blueprint
except Exception:
    get_swaggerui_blueprint = None


def _build_spec(base_url=""):
    """Return a minimal OpenAPI 3 spec describing the main endpoints."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "RPS Gesture API",
            "version": "1.0.0",
            "description": "API for Rock-Paper-Scissors gesture classification"
        },
        "servers": [{"url": base_url or "/"}],
        "paths": {
            "/api/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service health",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/predict": {
                "post": {
                    "summary": "Predict single image",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image": {"type": "string", "format": "binary"}
                                    },
                                    "required": ["image"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Prediction result", "content": {"application/json": {"schema": {"type": "object"}}}},
                        "400": {"description": "Bad request"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/api/upload": {
                "post": {
                    "summary": "Upload training images",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                                        "label": {"type": "string"}
                                    },
                                    "required": ["files", "label"]
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Upload result"}, "400": {"description": "Bad request"}}
                }
            },
            "/api/retrain": {
                "post": {
                    "summary": "Trigger retraining",
                    "responses": {"200": {"description": "Started"}, "400": {"description": "Already running"}}
                }
            },
            "/api/retrain/status": {
                "get": {"summary": "Retrain status", "responses": {"200": {"description": "Status"}}}
            },
            "/api/metrics": {
                "get": {"summary": "Get metrics", "responses": {"200": {"description": "Metrics list"}}}
            },
            "/api/uptime": {
                "get": {"summary": "Get uptime stats", "responses": {"200": {"description": "Uptime"}}}
            },
            "/api/stats": {
                "get": {"summary": "Get dataset stats", "responses": {"200": {"description": "Stats"}}}
            }
        }
    }
    return spec


def register_swagger(app, url_prefix=""):
    """Register `/openapi.json` and Swagger UI at `/docs` (or `/docs` + prefix).

    This function is safe if `flask_swagger_ui` is not installed; it will still
    register the JSON spec endpoint so clients can fetch it.
    """
    bp = Blueprint("openapi_bp", __name__)

    @bp.route("/openapi.json")
    def openapi_json():
        base_url = ""
        return jsonify(_build_spec(base_url=base_url))

    app.register_blueprint(bp)

    if get_swaggerui_blueprint:
        SWAGGER_URL = "/docs"
        API_URL = "/openapi.json"
        swaggerui_bp = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={
            "app_name": "RPS Gesture API"
        })
        app.register_blueprint(swaggerui_bp, url_prefix=SWAGGER_URL)
    else:
        # If the package isn't installed, we won't serve the static UI, but the
        # `/openapi.json` endpoint will still be available for external UIs.
        current_app.logger.info("flask_swagger_ui not available; only /openapi.json registered")
