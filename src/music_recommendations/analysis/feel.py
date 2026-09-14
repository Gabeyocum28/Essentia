"""The feel vector, as eight zero-shot questions asked in English.

The stack this replaced ran eleven trained classifier heads on a
Discogs-EffNet embedding. Those heads were non-commercially licensed, and
each one was a separate file to ship. CLAP gives the same thing for free:
its text tower
lands in the same 1024-d space as the audio tower, so "is this danceable?"
is the cosine between the track and the sentence "danceable, groovy,
rhythmic music".

A raw cosine is not usable as a score -- CLAP similarities sit in a narrow
band and shift with the phrasing of the prompt. Each axis is therefore a
*contrastive pair*: the score is the softmax over (positive, negative)
similarity, which cancels the common offset and gives a number in [0, 1]
that means "closer to this pole than that one". This is the standard CLIP /
CLAP zero-shot recipe, and it is why every entry below names both poles.

This module holds no torch import at module scope on purpose: callers that
only want FEEL_KEYS (the contract test, the server rendering axis labels)
must not pay for a 700 MB model.
"""
from __future__ import annotations

import threading

import numpy as np

__all__ = ["PROMPT_BANK", "FEEL_KEYS", "TEMPERATURE", "feel_scores",
           "text_matrix"]

# (name, positive prompt, negative prompt). Insertion order IS the feel-vector
# dimension order, and the names are what the web math panel labels its bars
# with. Changing a prompt changes the numbers for every track, so it is a
# FEATURES_VERSION bump (schema.py).
PROMPT_BANK: list[tuple[str, str, str]] = [
    ("energy",    "energetic, fast, intense music", "calm, slow, gentle music"),
    ("valence",   "happy, joyful, uplifting music", "sad, melancholic music"),
    ("tension",   "tense, dark, anxious music", "relaxed, peaceful music"),
    ("acoustic",  "acoustic instruments, unplugged",
                  "electronic, synthesized, produced"),
    ("danceable", "danceable, groovy, rhythmic music",
                  "music that is not for dancing"),
    ("vocal",     "a singer with vocals", "instrumental music without vocals"),
    ("bright",    "bright, crisp, high-frequency sound",
                  "dark, warm, muffled sound"),
    ("density",   "dense, busy, layered arrangement", "sparse, minimal arrangement"),
]

FEEL_KEYS = [name for name, _pos, _neg in PROMPT_BANK]

# CLAP's own logit scale (1 / temperature = 1 / 0.003 is ~333 in the config,
# but the trained logit_scale.exp() lands near 25 and that is what the
# wrapper's compute_similarity uses). 25 spreads a cosine gap of 0.04 -- a
# typical between-pole gap -- across roughly 0.27 of probability: enough to
# rank on, gentle enough that no axis saturates at 0 or 1 for ordinary music.
TEMPERATURE = 25.0

_text: np.ndarray | None = None
_text_lock = threading.Lock()


def text_matrix() -> np.ndarray:
    """(8, 2, 1024) prompt embeddings: [axis][0]=positive, [1]=negative.

    Computed once per process. Sixteen short prompts cost ~0.2 s once CLAP is
    loaded, but they are identical for every track ever analyzed, so paying
    that per group would be pure waste.
    """
    global _text
    if _text is not None:
        return _text
    with _text_lock:
        if _text is None:
            from . import clap

            prompts: list[str] = []
            for _name, pos, neg in PROMPT_BANK:
                prompts += [pos, neg]
            _text = clap.embed_text(prompts).reshape(len(PROMPT_BANK), 2, -1)
    return _text


def feel_scores(audio_emb: np.ndarray) -> np.ndarray:
    """(n, 8) in [0, 1]: for each axis, P(positive pole) given the track.

    `audio_emb` is (n, 1024) and expected L2-normalized (clap.embed_audio
    returns it that way), so the dot product is the cosine.
    """
    a = np.atleast_2d(np.asarray(audio_emb, dtype=np.float32))
    text = text_matrix()                          # (8, 2, 1024)
    sims = np.einsum("nd,apd->nap", a, text)      # (n, 8, 2) cosines
    logits = sims * TEMPERATURE
    logits -= logits.max(axis=2, keepdims=True)   # softmax, stably
    exp = np.exp(logits)
    probs = exp[:, :, 0] / exp.sum(axis=2)
    return probs.astype(np.float32)
