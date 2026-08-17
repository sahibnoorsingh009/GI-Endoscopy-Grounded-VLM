from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from .inference import AnalysisResult, GroundedGIService, grounded_answer
from .qwen_chat import QwenGIChatService


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "demo" / "examples" / "images"

WELCOME_HISTORY = [
    {
        "role": "assistant",
        "content": (
            "### Welcome to GI-EndoFM Chat\n"
            "Chat normally, or attach an endoscopy image when you want multimodal "
            "analysis with optional classifier and segmentation context."
        ),
    }
]


@lru_cache(maxsize=1)
def get_service() -> GroundedGIService:
    """Load the specialist classifier and segmenter once per process."""

    return GroundedGIService()


@lru_cache(maxsize=1)
def get_qwen_service() -> QwenGIChatService:
    """Load the selected native-Qwen LoRA once, on the first chat request."""

    return QwenGIChatService()


def image_fingerprint(image: Image.Image | np.ndarray | None) -> str | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def prediction_table(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "HyperKvasir category": item["display_label"],
                "Softmax score": round(float(item["softmax_score"]), 6),
            }
            for item in result.top_predictions
        ]
    )


def specialist_status(result: AnalysisResult, *, qwen_status: str) -> str:
    segmentation_status = result.evidence["segmentation"]["status"]
    if segmentation_status == "skipped_by_polyp_gate":
        segmentation_text = "Segmentation skipped by the polyp-family research gate."
    elif segmentation_status == "forced":
        segmentation_text = "Segmentation forced for research inspection."
    else:
        segmentation_text = "Gated segmentation completed."
    return (
        f"**{qwen_status}**  ·  SO400M classification completed  ·  "
        f"{segmentation_text}"
    )


def qwen_or_evidence_answer(
    *,
    image: Image.Image | np.ndarray | None,
    question: str,
    evidence: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    try:
        qwen_service = get_qwen_service()
        answer = qwen_service.answer(
            image=image,
            question=question,
            evidence=evidence,
            history=history,
        )
        return answer, qwen_service.runtime_label
    except Exception as error:
        if os.getenv("QWEN_ALLOW_SPECIALIST_FALLBACK", "1") != "1":
            raise
        if evidence is None:
            answer = (
                "The Qwen chat layer is currently unavailable "
                f"({type(error).__name__})."
            )
        else:
            fallback = grounded_answer(question, evidence)
            answer = (
                f"{fallback}\n\n"
                f"*The generative Qwen layer was unavailable ({type(error).__name__}); "
                "this response came from the deterministic specialist-evidence "
                "fallback.*"
            )
        return answer, "Specialist fallback active; Qwen unavailable"


def _run_specialists(
    image: Image.Image | np.ndarray,
    force_segmentation: bool,
) -> AnalysisResult:
    try:
        return get_service().analyze(
            image,
            force_segmentation=bool(force_segmentation),
        )
    except Exception as error:
        raise gr.Error(f"Specialist model analysis failed: {error}") from error


def analyze_image(
    image: Image.Image | np.ndarray | None,
    force_segmentation: bool,
):
    if image is None:
        raise gr.Error("Attach an endoscopy image first.")

    result = _run_specialists(image, force_segmentation)
    question = (
        "Describe this image naturally. When useful, relate your description to "
        "the optional classifier and segmentation outputs."
    )
    answer, qwen_status = qwen_or_evidence_answer(
        image=image,
        question=question,
        evidence=result.evidence,
        history=None,
    )
    # Display the initial image description, but do not turn it into a user/assistant
    # demonstration that biases all later conversational turns.
    history = [{"role": "assistant", "content": answer}]
    return (
        history,
        result.evidence,
        image_fingerprint(image),
        result.overlay,
        result.mask_image,
        prediction_table(result),
        result.evidence,
        specialist_status(result, qwen_status=qwen_status),
    )


def send_message(
    message: str,
    image: Image.Image | np.ndarray | None,
    force_segmentation: bool,
    evidence: dict[str, Any] | None,
    analyzed_image_fingerprint: str | None,
    history: list[dict[str, Any]] | None,
):
    question = (message or "").strip()
    if not question:
        raise gr.Error("Write a message first.")

    current_history = list(history or [])
    if image is None:
        answer, qwen_status = qwen_or_evidence_answer(
            image=None,
            question=question,
            evidence=None,
            history=current_history,
        )
        current_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return (
            current_history,
            "",
            None,
            None,
            None,
            None,
            pd.DataFrame(),
            None,
            f"**{qwen_status}**  ·  Text conversation; no image attached.",
        )

    current_fingerprint = image_fingerprint(image)
    image_changed = (
        evidence is None or current_fingerprint != analyzed_image_fingerprint
    )
    if image_changed:
        result = _run_specialists(image, force_segmentation)
        evidence = result.evidence
        # Prevent previous-image content from leaking into a new image conversation.
        model_history: list[dict[str, Any]] = []
    else:
        result = None
        model_history = current_history

    answer, qwen_status = qwen_or_evidence_answer(
        image=image,
        question=question,
        evidence=evidence,
        history=model_history,
    )
    if image_changed:
        current_history = []
    current_history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )

    if result is None:
        classification = pd.DataFrame(
            [
                {
                    "Rank": item["rank"],
                    "HyperKvasir category": item["display_label"],
                    "Softmax score": round(float(item["softmax_score"]), 6),
                }
                for item in evidence["classification"]["top_predictions"]
            ]
        )
        overlay = gr.skip()
        mask = gr.skip()
        status = f"**{qwen_status}**  ·  Reusing current specialist evidence."
    else:
        classification = prediction_table(result)
        overlay = result.overlay
        mask = result.mask_image
        status = specialist_status(result, qwen_status=qwen_status)

    return (
        current_history,
        "",
        evidence,
        current_fingerprint,
        overlay,
        mask,
        classification,
        evidence,
        status,
    )


def clear_chat():
    return (
        None,
        list(WELCOME_HISTORY),
        None,
        None,
        None,
        None,
        pd.DataFrame(),
        None,
        "**Models load lazily on the first image request.**",
        "",
    )


CSS = """
:root {
    --gi-bg: #0b0f14;
    --gi-panel: #111821;
    --gi-panel-soft: #17212d;
    --gi-border: #263343;
    --gi-text: #edf4f7;
    --gi-muted: #9db0bd;
    --gi-accent: #19c39c;
    --gi-accent-dark: #0f8f76;
}
body, .gradio-container {
    background: var(--gi-bg) !important;
    color: var(--gi-text) !important;
}
.gradio-container {
    max-width: 1720px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}
.app-shell { min-height: 100vh; }
.gi-sidebar {
    background: #0e141c;
    border-right: 1px solid var(--gi-border);
    min-height: 100vh;
    padding: 22px 18px !important;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}
.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #20d3aa, #1976a3);
    color: white;
    font-weight: 800;
    box-shadow: 0 10px 28px rgba(25,195,156,.22);
}
.brand h1 { font-size: 1.08rem; margin: 0; color: white; }
.brand p { font-size: .78rem; margin: 2px 0 0; color: var(--gi-muted); }
.model-card, .safety-card {
    padding: 13px 14px;
    border: 1px solid var(--gi-border);
    border-radius: 14px;
    background: var(--gi-panel);
    margin: 12px 0;
    font-size: .82rem;
    line-height: 1.45;
}
.model-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 99px;
    margin-right: 7px;
    background: var(--gi-accent);
    box-shadow: 0 0 12px rgba(25,195,156,.7);
}
.safety-card { border-color: #655622; background: #211d11; color: #e5d89f; }
.gi-main { min-height: 100vh; padding: 0 28px 24px !important; }
.topbar {
    position: sticky;
    top: 0;
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 70px;
    background: rgba(11,15,20,.92);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--gi-border);
    margin-bottom: 8px;
}
.topbar-title { color: white; font-weight: 650; }
.topbar-pill {
    padding: 7px 11px;
    border: 1px solid #265d53;
    border-radius: 999px;
    color: #9ce5d2;
    background: #10231f;
    font-size: .77rem;
}
.chat-window {
    border: 0 !important;
    background: transparent !important;
}
.chat-window .message {
    border-radius: 18px !important;
    border: 1px solid var(--gi-border) !important;
}
.composer {
    position: sticky;
    bottom: 12px;
    z-index: 4;
    padding: 10px !important;
    border: 1px solid #344454 !important;
    border-radius: 18px !important;
    background: #151e28 !important;
    box-shadow: 0 18px 44px rgba(0,0,0,.38);
}
.composer textarea { font-size: 1rem !important; }
.send-button {
    min-width: 92px !important;
    background: var(--gi-accent-dark) !important;
    border: 0 !important;
}
.quick-prompts button {
    border-radius: 999px !important;
    font-size: .8rem !important;
}
.status-line { color: var(--gi-muted); font-size: .82rem; }
footer { display: none !important; }
"""


example_images = [
    str(path)
    for path in sorted(EXAMPLE_DIRECTORY.glob("*"))
    if path.is_file()
]


with gr.Blocks(title="GI-EndoFM Chat", theme=gr.themes.Soft(), css=CSS) as demo:
    evidence_state = gr.State(value=None)
    image_fingerprint_state = gr.State(value=None)

    with gr.Row(elem_classes="app-shell", equal_height=False):
        with gr.Column(scale=3, min_width=290, elem_classes="gi-sidebar"):
            gr.HTML(
                """
                <div class="brand">
                    <div class="brand-mark">GI</div>
                    <div>
                        <h1>GI-EndoFM</h1>
                        <p>Endoscopy foundation-model research chat</p>
                    </div>
                </div>
                <div class="model-card">
                    <span class="model-dot"></span><strong>GI-EndoFM v0.1</strong><br>
                    Native Qwen3-VL 8B conversational backbone<br>
                    SO400M classifier + SigLIP2 segmentation
                </div>
                """
            )
            new_chat_button = gr.Button("＋ New chat", variant="secondary")
            image_input = gr.Image(
                type="pil",
                label="Attach endoscopy image",
                height=270,
            )
            force_segmentation = gr.Checkbox(
                value=False,
                label="Force segmentation",
                info="Research inspection only; normally controlled by the polyp gate.",
            )
            analyze_button = gr.Button(
                "Analyze attached image",
                variant="primary",
            )
            if example_images:
                gr.Examples(
                    examples=[[path] for path in example_images],
                    inputs=[image_input],
                    label="Research examples",
                )
            gr.HTML(
                """
                <div class="safety-card">
                    <strong>Research use only</strong><br>
                    Not validated for diagnosis, treatment, or patient management.
                    Do not upload identifiable patient information.
                </div>
                """
            )

        with gr.Column(scale=9, min_width=650, elem_classes="gi-main"):
            gr.HTML(
                """
                <div class="topbar">
                    <div class="topbar-title">Multimodal endoscopy conversation</div>
                    <div class="topbar-pill">Grounded model stack</div>
                </div>
                """
            )
            chatbot = gr.Chatbot(
                value=list(WELCOME_HISTORY),
                type="messages",
                height=545,
                show_label=False,
                elem_classes="chat-window",
            )

            with gr.Row(elem_classes="quick-prompts"):
                summarize_button = gr.Button("Summarize the outputs", size="sm")
                category_button = gr.Button("Explain the top categories", size="sm")
                location_button = gr.Button("Where is the predicted region?", size="sm")
                confidence_button = gr.Button("Explain the model scores", size="sm")

            with gr.Row(elem_classes="composer"):
                question_input = gr.Textbox(
                    placeholder="Message GI-EndoFM about the attached image…",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    container=False,
                    scale=12,
                )
                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_classes="send-button",
                )

            status_output = gr.Markdown(
                "**Models load lazily on the first image request.**",
                elem_classes="status-line",
            )

            with gr.Accordion("Model evidence and visual grounding", open=False):
                with gr.Row(equal_height=True):
                    overlay_output = gr.Image(
                        label="Conditional segmentation overlay",
                        height=300,
                    )
                    mask_output = gr.Image(
                        label="Binary mask",
                        height=300,
                    )
                classification_output = gr.Dataframe(
                    headers=["Rank", "HyperKvasir category", "Softmax score"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    wrap=True,
                    label="SO400M top-five evidence",
                )
                with gr.Accordion("Structured JSON evidence", open=False):
                    evidence_output = gr.JSON(label="Grounding payload")

    analysis_outputs = [
        chatbot,
        evidence_state,
        image_fingerprint_state,
        overlay_output,
        mask_output,
        classification_output,
        evidence_output,
        status_output,
    ]
    message_outputs = [
        chatbot,
        question_input,
        evidence_state,
        image_fingerprint_state,
        overlay_output,
        mask_output,
        classification_output,
        evidence_output,
        status_output,
    ]

    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input, force_segmentation],
        outputs=analysis_outputs,
        api_name="analyze_gi_image",
    )
    send_button.click(
        fn=send_message,
        inputs=[
            question_input,
            image_input,
            force_segmentation,
            evidence_state,
            image_fingerprint_state,
            chatbot,
        ],
        outputs=message_outputs,
        api_name="chat_with_gi_endofm",
    )
    question_input.submit(
        fn=send_message,
        inputs=[
            question_input,
            image_input,
            force_segmentation,
            evidence_state,
            image_fingerprint_state,
            chatbot,
        ],
        outputs=message_outputs,
    )
    new_chat_button.click(
        fn=clear_chat,
        outputs=[
            image_input,
            chatbot,
            evidence_state,
            image_fingerprint_state,
            overlay_output,
            mask_output,
            classification_output,
            evidence_output,
            status_output,
            question_input,
        ],
    )

    summarize_button.click(
        lambda: "Summarize the model outputs for this image.",
        outputs=question_input,
    )
    category_button.click(
        lambda: "Explain the top benchmark categories.",
        outputs=question_input,
    )
    location_button.click(
        lambda: "Where is the predicted segmentation region?",
        outputs=question_input,
    )
    confidence_button.click(
        lambda: "Explain the model scores and uncertainty.",
        outputs=question_input,
    )
