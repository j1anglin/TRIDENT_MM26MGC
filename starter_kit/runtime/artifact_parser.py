from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


MAPPING_PROMPT = """
You are an AI evaluation engine. Your task is to process an analysis of a digital image(`Analysis Text`) and determine which artifacts from a predefined list(`Artifact Definitions`) are present.

Your evaluation must be based ** strictly ** on the definitions provided.

Your output must be a simple key-value checklist suitable for automated parsing. Use "True" or "False". Do not include any justifications, explanations, or any text other than the artifact name and its corresponding boolean value.

---

# **1. Analysis Text**

{RESPONSE}

---

# **2. Artifact Definitions**

You must check for the presence of the following artifacts. An artifact is "True" ** only if ** the `Analysis Text` provides evidence that matches its specific `Definition`.

* **Blurriness**
    * **Definition**: ["The loss of sharpness and fine detail, making the image appear out of focus."]
* **Blockiness**
    * **Definition**: ["Visible square or rectangular patterns on the screen."]
* **Noise**
    * **Definition**: ["Random, fine speckles or a sandy texture across the image."]
* **Banding**
    * **Definition**: ["Distinct, abrupt steps or bands in areas that should have a smooth color gradient, like a sunset or a clear sky."]
* **Color Inconsistency**
    * **Definition**: ["Colors appear unnatural, with excessive saturation or vibrancy that makes the sample look too intense or unrealistic."]
* **Blending Artifacts**
    * **Definition**: ["Visible boundaries where elements should merge smoothly."]
* **Lighting Inconsistency**
    * **Definition**: ["Illumination that does not agree across the scene."]
* **Unnatural Texture**
    * **Definition**: ["The surface is overly smooth, missing the natural irregularities and tactile cues of real materials."]
* **Temporal Artifacts**
    * **Definition**: ["Inconsistencies across frames that break motion continuity."]
* **Flicker**
    * **Definition**: ["Noticeable and often rapid variation in the overall brightness of the video."]
* **Clipping**
    * **Definition**: ["A harsh, fuzzy, or crackling sound that occurs when the audio is too loud for the system to handle."]
* **Hiss**
    * **Definition**: ["High-frequency static noise, often described as a shhhh sound."]
* **Buzz**
    * **Definition**: ["Low-frequency tone, typically caused by electrical interference."]
* **Pops**
    * **Definition**: ["Abrupt, short, and sharp sounds that interrupt the audio."]
* **Reflection Inconsistency**
    * **Definition**: ["Reflections do not match the subject, lighting, or scene geometry."]
* **Shadow Inconsistency**
    * **Definition**: ["Shadows do not match the subject, lighting, or scene geometry."]
* **Spatial & Contact Incoherence**
    * **Definition**: ["Objects or people fail to make contact with surfaces or each other."]
* **Unrealistic Background**
    * **Definition**: ["Background lacks plausible detail, perspective, or depth."]
* **Anatomical Inconsistency**
    * **Definition**: ["Human anatomy is implausible."]
* **Unnatural Expressions**
    * **Definition**: ["Facial expressions do not align with emotion or context or appears unrealistic."]
* **Unnatural Gaze or Blinking**
    * **Definition**: ["Eye direction or blink behavior appears robotic."]
* **Unnatural Body or Head Movement**
    * **Definition**: ["Motion lacks physical plausibility."]
* **Object Integrity Flaws**
    * **Definition**: ["The object is incomplete, broken, or internally inconsistent."]
* **Unrecognizable Text**
    * **Definition**: ["The text is unrecognizable, incomplete, broken, or distorted."]
* **Unnatural Prosody**
    * **Definition**: ["Speech often sound robotic, monotonous, or flat, lacking natural intonation."]
* **Audio-Visual Desynchronization**
    * **Definition**: ["A mismatch between spoken audio and visible mouth movements or facial actions."]
* **Emotional Contradiction**
    * **Definition**: ["The face, voice, or body language conveys a different emotion than the content."]

---

# **Begin Evaluation**
"""

_LINE_RE = re.compile(
    r"^\s*[\-*]?\s*`?\s*\"?\s*(?P<key>[^:\n\r]+?)\s*\"?\s*`?\s*:\s*\"?\s*(?P<val>true|false)\s*\"?\s*,?\s*$",
    flags=re.IGNORECASE,
)


def _extract_artifacts(prompt: str) -> List[str]:
    matches = re.findall(r"^\*\s*\*\*(.*?)\*\*", prompt, flags=re.M)
    return [m.strip() for m in matches if m.strip()]


ARTIFACTS = _extract_artifacts(MAPPING_PROMPT)


def _normalize_artifact(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


_CANONICAL_BY_NORM = {_normalize_artifact(a): a for a in ARTIFACTS}


def _parse_boolish(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().strip("\"'").lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


def _extract_json_object_substring(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_artifact_map(response_text: str) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    text = (response_text or "").strip()
    if not text:
        return out

    for candidate in (text, _extract_json_object_substring(text) or ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            for key, value in obj.items():
                canonical = _CANONICAL_BY_NORM.get(_normalize_artifact(str(key)))
                if not canonical:
                    continue
                parsed = _parse_boolish(value)
                if parsed is None:
                    continue
                out[canonical] = parsed
            if out:
                return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        canonical = _CANONICAL_BY_NORM.get(_normalize_artifact(match.group("key").strip()))
        if not canonical:
            continue
        parsed = _parse_boolish(match.group("val").strip())
        if parsed is None:
            continue
        out[canonical] = parsed

    return out
