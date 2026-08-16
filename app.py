from __future__ import annotations

import os

import gradio as gr

from vlm_demo.app import CSS, demo


def _optional_basic_auth() -> tuple[str, str] | None:
    username = os.getenv("VLM_USERNAME")
    password = os.getenv("VLM_PASSWORD")
    if bool(username) != bool(password):
        raise RuntimeError(
            "Set both VLM_USERNAME and VLM_PASSWORD, or leave both unset."
        )
    return (username, password) if username and password else None


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))

    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        auth=_optional_basic_auth(),
    )
