import os


os.execvp(
    "gunicorn",
    [
        "gunicorn",
        "--bind",
        "0.0.0.0:8080",
        "--workers",
        "1",
        "--timeout",
        "60",
        "app:app",
    ],
)