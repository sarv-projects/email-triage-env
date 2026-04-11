"""
FastAPI application for the Email Triage Environment.

Endpoints:
    - POST /reset: Reset the environment and get the first email
    - POST /step: Submit a triage action and get the next email
    - GET /state: Get current environment state
    - GET /health: Health check
    - GET /: Service metadata
"""

import uvicorn

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_app

try:
    from models import EmailAction, EmailObservation
except ImportError:
    from ..models import EmailAction, EmailObservation

try:
    from .environment import EmailEnvironment
except ImportError:
    from environment import EmailEnvironment


app = create_app(
    EmailEnvironment,
    EmailAction,
    EmailObservation,
    env_name="email_triage_env",
)


@app.get("/")
def root():
    """Service metadata and endpoint listing."""
    return {
        "name": "Email Triage OpenEnv Environment",
        "version": "2.0.0",
        "description": (
            "Multi-step email triage environment where an AI agent processes "
            "a queue of real-world support emails, classifying category, urgency, "
            "and routing action. Supports 3 difficulty levels with 50+ email scenarios."
        ),
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
        "tasks": ["easy", "medium", "hard"],
    }


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
