from __future__ import annotations

import re
from typing import Dict, List, Pattern, Sequence


NEGATION_WORDS = {
    "no",
    "not",
    "without",
    "lack",
    "lacking",
    "absent",
    "free",
    "never",
    "neither",
    "nor",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "arent",
    "aren't",
    "cannot",
    "can't",
    "cant",
}

NEGATION_PHRASES: tuple[Pattern[str], ...] = (
    re.compile(r"\bno (?:clear )?(?:evidence|sign|indication|trace) of\b"),
    re.compile(r"\bfree of\b"),
    re.compile(r"\babsence of\b"),
    re.compile(r"\bwithout any\b"),
)


RULES: Dict[str, Sequence[str]] = {
    "Blurriness": [
        r"\bblur(?:red|ry|ring)?\b",
        r"\bout of focus\b",
        r"\bsoft focus\b",
        r"\bdefocus(?:ed)?\b",
        r"\bmotion blur\b",
        r"\bsmear(?:ed|ing)?\b",
    ],
    "Blockiness": [
        r"\bblock(?:y|iness)\b",
        r"\bmacroblock(?:ing)?\b",
        r"\bpixelat(?:ed|ion)\b",
        r"\bcompression artifacts?\b",
    ],
    "Noise": [
        r"\bgrain(?:y)?\b",
        r"\bspeckle(?:d|s)?\b",
        r"\bsandy\b",
        r"\bimage noise\b",
        r"\bvisual noise\b",
        r"\bsensor noise\b",
        r"\bnoisy (?:image|video|frame|visual)\b",
    ],
    "Banding": [
        r"\bbanding\b",
        r"\bposteriz(?:ed|ation)\b",
        r"\bcolor band(?:ing|s)?\b",
        r"\bgradient band(?:ing|s)?\b",
    ],
    "Color Inconsistency": [
        r"\bcolou?r (?:mismatch|shift|cast|inconsisten(?:t|cy))\b",
        r"\bunnatural colou?rs?\b",
        r"\bover[- ]saturated\b",
        r"\btoo (?:saturated|vibrant|intense)\b",
        r"\bsaturation (?:is )?(?:too high|excessive)\b",
        r"\bhue (?:shift|mismatch)\b",
        r"\bcolou?r tone (?:differs|mismatch)\b",
    ],
    "Blending Artifacts": [
        r"\bblending (?:artifact|issue|problem|seam)\b",
        r"\bblending\b",
        r"\bseam\b",
        r"\bedge (?:artifact|halo|fringing|fringe|outline)\b",
        r"\bhalo(?:ed|ing)?\b",
        r"\bcut[- ]out\b",
        r"\bcomposite(?:d|ing)?\b",
        r"\bmask(?:ing)?\b",
        r"\bmatte\b",
        r"\boverlay(?:ed)?\b",
        r"\bvisible (?:boundary|edge)\b",
        r"\bfeather(?:ed|ing)?\b",
    ],
    "Lighting Inconsistency": [
        r"\blighting (?:mismatch|inconsisten(?:t|cy))\b",
        r"\binconsistent lighting\b",
        r"\billumination (?:mismatch|inconsisten(?:t|cy))\b",
        r"\bwrong lighting\b",
        r"\blight source (?:does not|doesn't|doesnt|not) match\b",
        r"\bshading (?:mismatch|inconsisten(?:t|cy))\b",
        r"\buneven lighting\b",
    ],
    "Unnatural Texture": [
        r"\boverly smooth\b",
        r"\btoo smooth\b",
        r"\bairbrushed\b",
        r"\bplastic(?:-like)?\b",
        r"\bwaxy\b",
        r"\bmissing (?:fine )?texture\b",
        r"\blacks? (?:fine )?detail\b",
        r"\blacks? pores\b",
        r"\bno pores\b",
        r"\bunnatural texture\b",
        r"\bover[- ]smoothed\b",
        r"\bskin (?:looks )?smooth\b",
    ],
    "Temporal Artifacts": [
        r"\btemporal\b",
        r"\bframe[- ]to[- ]frame\b",
        r"\bframe[- ]by[- ]frame\b",
        r"\bacross frames\b",
        r"\bbetween frames\b",
        r"\bjitter(?:y)?\b",
        r"\bwobbl(?:e|y|ing)\b",
        r"\bshimmer(?:ing)?\b",
        r"\bswim(?:ming)?\b",
        r"\bdrift(?:ing)?\b",
        r"\binstabilit(?:y|ies)\b",
        r"\bwarping\b",
    ],
    "Flicker": [
        r"\bflicker(?:ing)?\b",
        r"\bstrob(?:e|ing)\b",
        r"\bbrightness (?:fluctuat(?:ion|ions|ing)|variation|changes?)\b",
        r"\bflash(?:ing)?\b",
    ],
    "Clipping": [
        r"\bclipp(?:ing|ed)\b",
        r"\boverdriven\b",
        r"\b(?:audio|sound|signal) distortion\b",
        r"\bcrackling\b",
    ],
    "Hiss": [
        r"\bhiss(?:ing)?\b",
        r"\bshhh\b",
        r"\bhigh[- ]frequency (?:noise|static)\b",
        r"\bwhite noise\b",
    ],
    "Buzz": [
        r"\bbuzz(?:ing)?\b",
        r"\bhum(?:ming)?\b",
        r"\blow[- ]frequency tone\b",
        r"\belectrical interference\b",
        r"\bground loop\b",
    ],
    "Pops": [
        r"\bpop(?:s|ping|ped)?\b",
        r"\bclick(?:s|ing|ed)?\b",
        r"\bcrackle\b",
    ],
    "Reflection Inconsistency": [
        r"\breflection(?:s)? (?:mismatch|inconsisten(?:t|cy)|do not match|does not match|don't match)\b",
        r"\bwrong reflection\b",
        r"\breflections? (?:are )?incorrect\b",
    ],
    "Shadow Inconsistency": [
        r"\bshadow(?:s)? (?:mismatch|inconsisten(?:t|cy)|do not match|does not match|don't match)\b",
        r"\bwrong shadow\b",
        r"\bmissing shadow\b",
        r"\bshadow (?:direction|shape) (?:is )?(?:wrong|off|inconsistent)\b",
    ],
    "Spatial & Contact Incoherence": [
        r"\bfloating\b",
        r"\bhover(?:ing)?\b",
        r"\bnot (?:touching|connected|contact)\b",
        r"\bno contact\b",
        r"\black of contact\b",
        r"\bintersect(?:ing|ion)\b",
        r"\binterpenetrat(?:e|ion|ing)\b",
        r"\bpassing through\b",
        r"\bdetached\b",
        r"\bmisaligned\b",
    ],
    "Unrealistic Background": [
        r"\bunrealistic background\b",
        r"\bbackground (?:looks|seems) (?:fake|unrealistic)\b",
        r"\bbackground lacks (?:detail|depth|perspective)\b",
        r"\bflat background\b",
        r"\bstatic background\b",
        r"\bbackground is (?:a )?still image\b",
        r"\bbackground (?:does not|doesn't) move\b",
    ],
    "Anatomical Inconsistency": [
        r"\banatom(?:y|ical)\b",
        r"\bextra (?:finger|fingers|limb|arms|legs)\b",
        r"\bmissing (?:finger|fingers|limb|arm|leg)\b",
        r"\bwrong number of (?:fingers|limbs)\b",
        r"\bimplausible (?:body|pose|anatomy)\b",
        r"\bdeformed (?:hand|face|body|limb)\b",
        r"\bmalformed\b",
        r"\bdisproportionate\b",
    ],
    "Unnatural Expressions": [
        r"\bunnatural expression(?:s)?\b",
        r"\bstiff expression(?:s)?\b",
        r"\bexpression (?:is )?unnatural\b",
        r"\bfrozen expression\b",
        r"\bblank expression\b",
    ],
    "Unnatural Gaze or Blinking": [
        r"\bgaze\b",
        r"\bstare(?:ing)?\b",
        r"\bblink(?:ing|s|ed)?\b",
        r"\bno blinking\b",
        r"\beye movement(?:s)? (?:are )?(?:unnatural|robotic)\b",
        r"\beyes (?:do not|don't) move\b",
        r"\bglassy eyes\b",
    ],
    "Unnatural Body or Head Movement": [
        r"\bhead movement(?:s)?\b",
        r"\bbody movement(?:s)?\b",
        r"\bunnatural motion\b",
        r"\bunnatural movement\b",
        r"\bjerky\b",
        r"\brig(?:id|idity)\b",
        r"\bstiff\b",
        r"\bpuppet(?:-like)?\b",
        r"\brobotic movement\b",
    ],
    "Object Integrity Flaws": [
        r"\bobject (?:is )?(?:broken|incomplete|damaged)\b",
        r"\bmissing parts\b",
        r"\bdeformed\b",
        r"\bwarped\b",
        r"\bmelting\b",
        r"\bglitch(?:y)?\b",
        r"\binternal inconsisten(?:t|cy)\b",
        r"\bobject integrity\b",
    ],
    "Unrecognizable Text": [
        r"\btext (?:is )?(?:unreadable|illegible|unrecognizable|distorted|garbled|broken)\b",
        r"\bunreadable text\b",
        r"\billegible text\b",
        r"\bgarbled text\b",
        r"\bgibberish text\b",
    ],
    "Unnatural Prosody": [
        r"\bunnatural prosody\b",
        r"\bmonoton(?:ous|y)\b",
        r"\bmonotone\b",
        r"\bflat intonation\b",
        r"\bflat cadence\b",
        r"\brobotic (?:voice|speech)\b",
        r"\btext[- ]to[- ]speech\b",
        r"\btts\b",
        r"\bcadence (?:is )?(?:flat|even|robotic|unnatural)\b",
        r"\bprosody\b",
        r"\bsynthetic voice\b",
    ],
    "Audio-Visual Desynchronization": [
        r"\blip[- ]?sync\b",
        r"\b(?:lip|mouth) (?:movement|movements) (?:do not|does not|don't) match\b",
        r"\bout of sync\b",
        r"\bnot synchronized\b",
        r"\baudio[- ]visual (?:mismatch|desync|desynchronization)\b",
        r"\bdesynchroni(?:s|z)ation\b",
        r"\bav sync\b",
    ],
    "Emotional Contradiction": [
        r"\bemotional contradiction\b",
        r"\bemotion (?:mismatch|incongruent)\b",
        r"\bcontradict(?:s|ing)? (?:emotion|tone|expression|body language)\b",
        r"\btone (?:does not|doesn't|not) match\b",
        r"\bexpression (?:does not|doesn't|not) match\b",
        r"\bbody language (?:does not|doesn't|not) match\b",
        r"\bsmiling while\b",
    ],
}


COMPILED_RULES: Dict[str, List[Pattern[str]]] = {
    name: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for name, patterns in RULES.items()
}


def _is_negated(text: str, match_start: int, *, window_chars: int = 80, window_words: int = 6) -> bool:
    if match_start <= 0:
        return False
    window = text[max(0, match_start - window_chars) : match_start]
    for phrase in NEGATION_PHRASES:
        if phrase.search(window):
            return True
    words = re.findall(r"[a-zA-Z']+", window)
    if not words:
        return False
    return any(word in NEGATION_WORDS for word in words[-window_words:])


def _pattern_hit(text: str, pattern: Pattern[str]) -> bool:
    for match in pattern.finditer(text):
        if not _is_negated(text, match.start()):
            return True
    return False


def map_analysis_text(text: str, artifacts: Sequence[str]) -> Dict[str, bool]:
    lowered = str(text or "").lower()
    results: Dict[str, bool] = {}
    for artifact in artifacts:
        patterns = COMPILED_RULES.get(artifact, [])
        results[artifact] = any(_pattern_hit(lowered, pattern) for pattern in patterns)
    if results.get("Flicker") and "Temporal Artifacts" in results:
        results["Temporal Artifacts"] = True
    return results
