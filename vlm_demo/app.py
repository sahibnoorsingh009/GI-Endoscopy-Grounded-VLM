from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from .inference_new import GroundedGIService, grounded_answer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "demo" / "examples" / "images"


@lru_cache(maxsize=1)
def get_service() -> GroundedGIService:
    return GroundedGIService()


def analyze_image(image, force_segmentation: bool):
    if image is None:
        raise gr.Error("Upload an endoscopy image first.")

    try:
        result = get_service().analyze(
            image,
            force_segmentation=bool(force_segmentation),
        )
    except Exception as error:
        raise gr.Error(f"Model analysis failed: {error}") from error

    prediction_table = pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "HyperKvasir category": item["display_label"],
                "Softmax score": round(float(item["softmax_score"]), 6),
            }
            for item in result.top_predictions
        ]
    )
    summary = grounded_answer("Summarize the model outputs.", result.evidence)
    history = [{"role": "assistant", "content": summary}]

    segmentation_status = result.evidence["segmentation"]["status"]
    if segmentation_status == "skipped_by_polyp_gate":
        status = (
            "Segmentation was safely skipped by the polyp gate. Select "
            "**Force segmentation** only for controlled research inspection."
        )
    elif segmentation_status == "forced":
        status = (
            "Segmentation was forced regardless of the classification gate; interpret "
            "the mask as an exploratory output."
        )
    else:
        status = "Classification and gated segmentation completed."

    return (
        result.original,
        result.overlay,
        result.mask_image,
        prediction_table,
        result.evidence,
        history,
        result.evidence,
        status,
    )


def ask_about_result(question: str, evidence, history):
    if not evidence:
        raise gr.Error("Run image analysis before asking a question.")
    clean_question = (question or "").strip()
    if not clean_question:
        raise gr.Error("Enter a question about the model output.")

    answer = grounded_answer(clean_question, evidence)
    updated_history = list(history or [])
    updated_history.extend(
        [
            {"role": "user", "content": clean_question},
            {"role": "assistant", "content": answer},
        ]
    )
    return updated_history, ""


def clear_analysis():
    return None, None, None, None, None, None, [], None, ""


CSS = """
.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}
.hero {
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 16px;
    color: white;
    background: linear-gradient(135deg, #102942 0%, #17637b 55%, #15836e 100%);
    box-shadow: 0 16px 35px rgba(0, 0, 0, 0.2);
}
.hero h1 { margin: 0; color: white; font-size: 2rem; }
.hero p { margin: 8px 0 0; color: #e4f7ff; font-size: 1rem; }
.notice {
    padding: 12px 16px;
    margin-bottom: 18px;
    border: 1px solid #d3ac39;
    border-radius: 12px;
    background: #fff7d7;
    color: #5c4700;
}
.model-note {
    padding: 12px 16px;
    border-left: 4px solid #16866c;
    border-radius: 8px;
    background: rgba(22, 134, 108, 0.08);
}
footer { display: none !important; }
"""


example_images = [
    str(path)
    for path in sorted(EXAMPLE_DIRECTORY.glob("*"))
    if path.is_file()
]


with gr.Blocks(title="GI Endoscopy Grounded VLM Interface") as demo:
    evidence_state = gr.State(value=None)

    gr.HTML(
        """
        <div class="hero">
            <h1>GI Endoscopy Grounded VLM Interface</h1>
            <p>
                SO400M-384 classification + seed-43 SigLIP2 polyp segmentation +
                evidence-grounded question answering
            </p>
        </div>
        """
    )
    gr.HTML(
        """
        <div class="notice">
            <strong>Research demonstration only.</strong>
            Not validated for diagnosis, treatment, or patient management. Do not
            upload identifiable patient information. Model scores are not calibrated
            clinical probabilities.
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            image_input = gr.Image(
                type="pil",
                label="Endoscopy image",
                height=360,
            )
            force_segmentation = gr.Checkbox(
                value=False,
                label="Force segmentation (research inspection only)",
                info=(
                    "Normally the Kvasir-SEG branch runs only when the classifier's "
                    "combined polyp-family score passes the gate."
                ),
            )
            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze image",
                    variant="primary",
                    size="lg",
                )
                clear_button = gr.Button("Clear")

            if example_images:
                gr.Examples(
                    examples=[[path] for path in example_images],
                    inputs=[image_input],
                    label="Kvasir-SEG examples",
                )

            gr.HTML(
                """
                <div class="model-note">
                    <strong>Why two branches?</strong><br>
                    The classification model is strongest for the 23-class semantic
                    task. The existing full SigLIP2 seed-43 model is strongest for
                    pixel localization. A shared encoder is an optional experiment,
                    not a prerequisite for this interface.
                </div>
                """
            )

        with gr.Column(scale=8):
            with gr.Row(equal_height=True):
                original_output = gr.Image(
                    label="Analyzed image",
                    height=330,
                )
                overlay_output = gr.Image(
                    label="Conditional polyp overlay",
                    height=330,
                )
            mask_output = gr.Image(
                label="Conditional binary mask",
                height=300,
            )
            status_output = gr.Markdown()

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown("## Classification evidence")
            classification_output = gr.Dataframe(
                headers=["Rank", "HyperKvasir category", "Softmax score"],
                datatype=["number", "str", "number"],
                interactive=False,
                wrap=True,
                label="Top five categories",
            )
            with gr.Accordion("Structured evidence", open=False):
                evidence_output = gr.JSON(label="Model evidence")

        with gr.Column(scale=7):
            gr.Markdown("## Ask about the model outputs")
            chatbot = gr.Chatbot(
                type="messages",
                height=390,
                label="Grounded assistant",
            )
            question_input = gr.Textbox(
                label="Question",
                placeholder=(
                    "Examples: What is the top category? Where is the predicted "
                    "region? How confident is the classifier?"
                ),
                lines=2,
            )
            ask_button = gr.Button("Ask from evidence")

    gr.Markdown(
        "[Classification code](https://github.com/sahibnoorsingh009/GI-SigLIP2-DINO-HyperKvasir)"
        " · [Segmentation code](https://github.com/sahibnoorsingh009/Kvasir-SigLIP-Segmentation)"
        " · The language layer is deterministic and may only use the displayed specialist-model evidence."
    )

    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input, force_segmentation],
        outputs=[
            original_output,
            overlay_output,
            mask_output,
            classification_output,
            evidence_output,
            chatbot,
            evidence_state,
            status_output,
        ],
        api_name="analyze_gi_image",
    )
    ask_button.click(
        fn=ask_about_result,
        inputs=[question_input, evidence_state, chatbot],
        outputs=[chatbot, question_input],
        api_name="ask_from_evidence",
    )
    question_input.submit(
        fn=ask_about_result,
        inputs=[question_input, evidence_state, chatbot],
        outputs=[chatbot, question_input],
    )
    clear_button.click(
        fn=clear_analysis,
        outputs=[
            image_input,
            original_output,
            overlay_output,
            mask_output,
            classification_output,
            evidence_output,
            chatbot,
            evidence_state,
            status_output,
        ],
    )
