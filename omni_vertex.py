"""Gemini Omni video on Vertex AI.

Omni is not served through generate_videos like Veo. It uses Google's
Interactions API, called here directly with the service account's OAuth token.

Omni accepts most bad values at submit and only reports them when the job
fails, so check_params refuses them before anything is sent.
"""

import asyncio
import base64
import io
import json
import time

import comfy.model_management
import numpy as np
import requests
from PIL import Image
from comfy_api.latest import io as cio
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from .veo_vertex import GoogleVeoVertexVideoGenerator
from .vertex_async import run_blocking
from .vertex_credentials import (
    CREDENTIALS_TOOLTIP,
    default_credentials,
    list_credentials,
    resolve_credentials,
)

# Model page and video generation guides, checked 2026-11-02.
MODEL_RESOLUTIONS = {
    "gemini-omni-1.1-flash-preview": ("360p", "720p", "1080p", "4k"),
    "gemini-omni-flash-preview": ("720p",),
}
# The image-to-video and first/last frame guides list only this model.
FRAME_MODELS = ("gemini-omni-1.1-flash-preview",)
RESOLUTIONS = ["360p", "720p", "1080p", "4k"]
ASPECT_RATIOS = ["16:9", "9:16"]
MIN_DURATION, MAX_DURATION = 3, 10
MAX_REFERENCES = 10
MAX_INPUT_BYTES = 20 * 1024 * 1024

INTERACTIONS_URL = "https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/interactions"
POLL_SECONDS = 5
TIMEOUT_SECONDS = 1200


def check_params(model, resolution, aspect_ratio, duration, prompt, first_frame, last_frame, reference_count):
    if resolution not in MODEL_RESOLUTIONS[model]:
        raise ValueError(f"{model} supports {', '.join(MODEL_RESOLUTIONS[model])}, not {resolution}.")
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Omni supports aspect ratios {' and '.join(ASPECT_RATIOS)}, not {aspect_ratio}.")
    if not MIN_DURATION <= duration <= MAX_DURATION:
        raise ValueError(f"Omni supports durations of {MIN_DURATION}-{MAX_DURATION} seconds, not {duration}.")
    if not prompt.strip() and first_frame is None:
        raise ValueError("A prompt or a first frame must be provided.")
    if (first_frame is not None or last_frame is not None) and model not in FRAME_MODELS:
        raise ValueError(f"{model} does not support first or last frames. Use {FRAME_MODELS[0]}.")
    if last_frame is not None and first_frame is None:
        raise ValueError("Omni needs a first frame when a last frame is given.")
    if reference_count and (first_frame is not None or last_frame is not None):
        raise ValueError("Omni treats references and first/last frames as separate tasks. Use one or the other.")
    if reference_count > MAX_REFERENCES:
        raise ValueError(f"Omni accepts at most {MAX_REFERENCES} images, but {reference_count} are connected.")


def reference_frames(references):
    """Every frame of every connected slot, in slot order. Each frame is one image."""
    frames = []
    for name in sorted(references, key=lambda n: int(n.rsplit("_", 1)[1])):
        tensor = references[name]
        if tensor is not None:
            frames.extend(tensor if tensor.dim() == 4 else tensor.unsqueeze(0))
    return frames


def image_part(tensor, label, aspect_ratio=None):
    """An inline PNG input for the request. A batch sends its first frame, as Veo does."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"{label} is {len(data) / 1024 / 1024:.1f} MB as PNG; Omni accepts at most 20 MB per image.")
    if aspect_ratio:
        want_w, want_h = (int(x) for x in aspect_ratio.split(":"))
        if abs((pil.width / pil.height) / (want_w / want_h) - 1) > 0.02:
            print(f"[VertexAPI] Omni: {label} is {pil.width}x{pil.height}, not {aspect_ratio}. "
                  f"The model has to fit it to that shape.")
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": "image/png"}


def build_request(prompt, model, resolution, aspect_ratio, duration, first_frame, last_frame, references):
    images = []
    if first_frame is not None:
        images.append(image_part(first_frame, "first_frame", aspect_ratio))
    if last_frame is not None:
        images.append(image_part(last_frame, "last_frame", aspect_ratio))
    images += [image_part(r, f"reference {i}") for i, r in enumerate(references, 1)]

    body = {
        "model": model,
        "background": True,
        "input": ([{"type": "text", "text": prompt}] if prompt.strip() else []) + images,
        "response_format": [{"type": "video", "aspect_ratio": aspect_ratio, "resolution": resolution,
                             "duration": f"{duration}s"}],
    }
    if first_frame is not None:
        body["generation_config"] = {"video_config": {"task": "image_to_video"}}
    elif references:
        body["generation_config"] = {"video_config": {"task": "reference_to_video"}}
    return body


def make_session(service_account_json):
    try:
        sa_info = json.loads(service_account_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {str(e)}")
    return AuthorizedSession(service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    ))


def http_error(status, text):
    head = f"HTTP {status}: {text[:600]}"
    if status in (401, 403):
        return RuntimeError(f"Vertex refused the request ({head}). Check that the service account has "
                            f"roles/aiplatform.user and that the Vertex AI API is enabled for the project.")
    if status == 429:
        return RuntimeError(f"Quota / rate limit hit ({head}). Wait and retry, or request more quota for Omni.")
    return RuntimeError(f"Omni API Error: {head}")


async def wait_for_interaction(session, url, name):
    """Poll a submitted job until it finishes. Google keeps running and billing it
    whatever happens here, so errors name the job."""
    t0 = time.time()
    while True:
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except comfy.model_management.InterruptProcessingException:
            print(f"[VertexAPI] Omni: stopped waiting for {name}. The job is still running at Google "
                  f"and will still be charged.")
            raise
        try:
            r = await run_blocking(session.get, f"{url}/{name}", timeout=300)
        except requests.RequestException as e:
            print(f"[VertexAPI] Omni: temporary network error while waiting ({type(e).__name__}); still waiting.")
            r = None
        if r is not None and r.status_code == 200:
            interaction = r.json()
            if interaction.get("status") != "in_progress":
                return interaction
        elif r is not None and r.status_code < 500 and r.status_code != 429:
            raise http_error(r.status_code, r.text)
        if time.time() - t0 > TIMEOUT_SECONDS:
            raise RuntimeError(f"Omni job {name} was still running after {TIMEOUT_SECONDS}s. "
                               f"Queueing again submits and pays for a new job.")
        # asyncio.sleep, not time.sleep: the job runs for minutes and must not
        # hold the event loop while it polls.
        await asyncio.sleep(POLL_SECONDS)


def video_bytes(interaction):
    """MP4 bytes from a finished interaction, or the reason there are none."""
    errors = interaction.get("errors") or []
    if interaction.get("status") != "completed" or errors:
        detail = "; ".join(str(e.get("message", e)) for e in errors) or f"status={interaction.get('status')}"
        raise RuntimeError(f"Omni job {interaction.get('id')} failed: {detail[:800]}")
    videos, texts, shape = [], [], []
    for step in interaction.get("steps") or []:
        items = step.get("content") or []
        shape.append((step.get("type"), [item.get("type") for item in items]))
        if step.get("type") != "model_output":
            continue
        for item in items:
            if item.get("type") == "video" and item.get("data"):
                videos.append(base64.b64decode(item["data"]))
            elif item.get("type") == "text" and item.get("text"):
                texts.append(item["text"])
    if not videos:
        said = f': "{" ".join(texts)[:400]}"' if texts else ""
        # The fields, never their contents: enough to see a changed response format.
        raise RuntimeError(f"Omni job {interaction.get('id')} finished with no video{said} "
                           f"(fields: {sorted(interaction)}; steps: {shape}).")
    return videos[0]


class GeminiOmniVertex(cio.ComfyNode):

    @classmethod
    def define_schema(cls):
        return cio.Schema(
            node_id="GeminiOmniVertex",
            display_name="Gemini Omni (Vertex AI)",
            category="video/generation",
            description="Generate one video, always with audio, with Gemini Omni on Vertex AI.",
            inputs=[
                # Fixed sockets first, the growing reference list last: ComfyUI restores
                # links by slot position, so anything after it would collide with slots
                # that grow later.
                cio.String.Input("project_id", force_input=True, optional=True,
                                 tooltip="Overrides the credentials file."),
                cio.String.Input("service_account", force_input=True, optional=True,
                                 tooltip="Service account JSON. Overrides the credentials file."),
                cio.Image.Input("first_frame", optional=True,
                                tooltip="gemini-omni-1.1-flash-preview only. Say how to use it in the prompt, "
                                        "for example \"Use the image as the first frame.\" "
                                        "A batch sends its first frame."),
                cio.Image.Input("last_frame", optional=True,
                                tooltip="gemini-omni-1.1-flash-preview only, and needs a first frame. "
                                        "Say how to use it in the prompt. A batch sends its first frame."),
                cio.Autogrow.Input(
                    "references",
                    optional=True,
                    template=cio.Autogrow.TemplateNames(
                        cio.Image.Input("reference"),
                        names=[f"reference_{i}" for i in range(1, MAX_REFERENCES + 1)],
                        min=0,
                    ),
                    tooltip="Reference images. A new slot appears as you connect one. Up to 10; every frame "
                            "of a batch counts. Cannot be combined with first/last frames. End the prompt "
                            "with an instruction such as \"Use the image(s) as references to generate the video.\"",
                ),
                cio.String.Input("prompt", multiline=True, default=""),
                cio.Combo.Input("credentials", options=list_credentials(), default=default_credentials(),
                                tooltip=CREDENTIALS_TOOLTIP),
                cio.Combo.Input("model", options=list(MODEL_RESOLUTIONS), default="gemini-omni-1.1-flash-preview",
                                tooltip="gemini-omni-flash-preview is 720p only, with no first or last frames."),
                cio.Combo.Input("resolution", options=RESOLUTIONS, default="720p",
                                tooltip="360p, 1080p and 4k are gemini-omni-1.1-flash-preview only."),
                cio.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, default="16:9"),
                cio.Int.Input("duration_seconds", default=4, min=MIN_DURATION, max=MAX_DURATION, step=1,
                              tooltip="3 to 10 seconds."),
            ],
            outputs=[
                cio.Image.Output("frames", display_name="frames"),
                cio.Audio.Output("audio", display_name="audio"),
            ],
        )

    @classmethod
    async def execute(cls, prompt, credentials, model, resolution, aspect_ratio, duration_seconds,
                      first_frame=None, last_frame=None, references=None,
                      project_id=None, service_account=None) -> cio.NodeOutput:
        reference_list = reference_frames(references or {})
        check_params(model, resolution, aspect_ratio, duration_seconds, prompt, first_frame, last_frame,
                     len(reference_list))

        project_id, service_account = resolve_credentials(credentials, project_id, service_account)
        session = make_session(service_account)
        url = INTERACTIONS_URL % project_id.strip()

        body = await run_blocking(build_request, prompt, model, resolution, aspect_ratio, duration_seconds,
                                  first_frame, last_frame, reference_list)
        r = await run_blocking(session.post, url, json=body, timeout=300)
        if r.status_code != 200:
            raise http_error(r.status_code, r.text)
        name = r.json()["id"]
        print(f"[VertexAPI] Omni job {name}: {model}, {resolution}, {aspect_ratio}, {duration_seconds}s.")

        interaction = await wait_for_interaction(session, url, name)
        data = video_bytes(interaction)
        frames, audio = await run_blocking(GoogleVeoVertexVideoGenerator._decode, data, True)
        return cio.NodeOutput(frames, audio)


NODE_CLASS_MAPPINGS = {"GeminiOmniVertex": GeminiOmniVertex}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiOmniVertex": "Gemini Omni (Vertex AI)"}
