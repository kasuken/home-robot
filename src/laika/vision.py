"""Vision requests: one captured frame to a vision model, a validated Observation back.

`VisionDescriber` is the `Observer`'s `describe` callable. It sends a single JPEG
with a strict-JSON prompt through the provider-independent `ChatClient`
(stateless `complete()`, no conversation history) and validates the reply with
`parse_model_observation`. Frames leave the robot one at a time and only to the
configured endpoint; there is never video.

Failure handling matches the conversation: an unreachable endpoint is reported to
the shared `AIEndpointMonitor`, so an outage seen by vision also stops movement.
A malformed reply raises `InvalidObservation`, which the `Observer` logs and
records once. Oversized frames and frames over the per-minute cap are skipped,
never queued. Pillow is optional: when installed, frames are shrunk to fit.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from laika.ai_client import AIUnavailable, ChatClient, image_message, system_message
from laika.config import AIConfig, VisionConfig
from laika.events import AIEndpointMonitor
from laika.observations import Observation, parse_model_observation
from laika.robot import ImageObservation

log = logging.getLogger(__name__)

RATE_WINDOW_S = 60.0
MAX_PROMPT_OBJECTS = 15  # below the validator limit, so a reply that obeys always passes
# Descriptions are factual: little randomness keeps the JSON well formed.
TEMPERATURE = 0.2
# Fallback qualities tried when a downscaled frame is still over the size limit.
_QUALITY_STEPS = (70, 55, 40)

SYSTEM_PROMPT = f"""You are the eyes of Laika, a small home robot. You receive one camera frame \
and describe what is actually visible in it, in a household context.

Reply with STRICT JSON only: no markdown, no code fences, no text outside the object. Schema:
{{"description": "one short English sentence",
 "objects": [{{"name": "lowercase English noun", "confidence": 0.0-1.0, "position": "left" | "centre" | "right"}}],
 "people": whole number,
 "confidence": 0.0-1.0}}

Rules:
- List at most {MAX_PROMPT_OBJECTS} objects, only things you can really see; never guess what is out of view.
- "confidence" is how sure you are of the whole description; lower it when the image is dark, blurry or unclear.
- People: only count them. Never identify anyone, never name anyone, never describe faces or other \
identifying features.
- If you see nothing recognisable, say so in the description and return an empty objects list."""


def user_prompt(location: str | None) -> str:
    where = f" The robot is in the {location}." if location else ""
    return f"Describe this camera frame.{where} Reply with the JSON object only."


def build_vision_client(ai: AIConfig, vision: VisionConfig) -> ChatClient:
    """A separate client for vision: the vision model and timeout, same provider and credentials."""
    model = vision.model or ai.model
    return ChatClient(
        base_url=ai.base_url,
        api_key=ai.api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        timeout_s=vision.timeout_s,
        temperature=TEMPERATURE,
        provider=ai.provider,
        azure_deployment=(vision.model or ai.azure_deployment) if ai.provider == "azure" else "",
        azure_api_version=ai.azure_api_version,
    )


def shrink_jpeg(jpeg: bytes, max_side_px: int, quality: int, max_bytes: int) -> bytes | None:
    """Re-encode `jpeg` to fit the limits, or None when Pillow is missing or cannot read it (the caller then uses the original).

    Pillow is imported here, lazily: it ships with Raspberry Pi OS (picamera2 needs it)
    but is not a dependency of Laika. A frame that already fits is returned untouched.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(jpeg)) as opened:
            opened.load()
            picture = opened.convert("RGB")
        resized = max(picture.size) > max_side_px
        if resized:
            picture.thumbnail((max_side_px, max_side_px))
        elif len(jpeg) <= max_bytes:
            return jpeg
        for q in (quality, *(s for s in _QUALITY_STEPS if s < quality)):
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=q)
            data = buffer.getvalue()
            if len(data) <= max_bytes:
                return data
        return data
    except Exception as error:  # noqa: BLE001 - the size limit still applies to the original
        log.debug("vision: cannot re-encode the frame, sending it as is: %s", error)
        return None


class VisionDescriber:
    """Callable `(ImageObservation, location) -> Observation | None` for `Observer(describe=...)`."""

    def __init__(
        self,
        client: ChatClient,
        config: VisionConfig,
        monitor: AIEndpointMonitor | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.config = config
        self.monitor = monitor
        self._clock = clock
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()
        self.skipped = 0

    def __call__(self, image: ImageObservation, location: str | None) -> Observation | None:
        jpeg = self._prepare(image)
        if jpeg is None or not self._allow():
            return None
        messages: list[dict[str, Any]] = [
            system_message(SYSTEM_PROMPT),
            image_message(user_prompt(location), jpeg, detail=self.config.detail),
        ]
        try:
            turn = self.client.complete(messages, json_mode=True)
        except AIUnavailable as error:
            if error.rejected:
                # The endpoint is up but refused the frame (often a model without image
                # support): a settings problem, so it must not stop driving like an outage.
                log.warning("vision request rejected, check [vision] model supports images: %s", error)
                return None
            log.warning("vision request failed: %s", error)
            if self.monitor is not None:
                self.monitor.failed(error)
            return None
        if self.monitor is not None:
            self.monitor.succeeded()
        # InvalidObservation is not caught: the Observer logs and records it once.
        return parse_model_observation(turn.content, image=image, location=location)

    def _prepare(self, image: ImageObservation) -> bytes | None:
        """The bytes to send: the frame, shrunk if Pillow is there and it helps; None to skip."""
        jpeg = image.jpeg
        if not jpeg:
            self._skip("the frame has no JPEG")
            return None
        limit = self.config.max_image_kb * 1024
        shrunk = shrink_jpeg(jpeg, self.config.max_side_px, self.config.jpeg_quality, limit)
        if shrunk is not None:
            jpeg = shrunk
        if len(jpeg) > limit:
            self._skip(f"the frame is {len(jpeg) // 1024} KB, over max_image_kb {self.config.max_image_kb}")
            return None
        return jpeg

    def _allow(self) -> bool:
        """Sliding one-minute window; a frame over the cap is skipped, never queued."""
        with self._lock:
            now = self._clock()
            while self._sent and now - self._sent[0] >= RATE_WINDOW_S:
                self._sent.popleft()
            if len(self._sent) >= self.config.max_per_minute:
                self._skip(f"over max_per_minute {self.config.max_per_minute}")
                return False
            self._sent.append(now)
            return True

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        log.warning("vision: frame not sent, %s", reason)
