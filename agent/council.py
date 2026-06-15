"""Council — two providers debate a question, a neutral pass synthesizes consensus.

Depersonalized and provider-pluggable: no built-in personas. Put a strong model in
one seat and a cheap one in the other (e.g. Claude proposes, DeepSeek critiques), or
the same model in both. One seat proposes a position; the other critiques on the
merits or concedes; after a few rounds an impartial "secretary" pass distills points
of agreement, residual disagreements, and a recommendation.

This reuses only `Provider.chat` — it is independent of tools, memory, and workdir.
"""
from __future__ import annotations

from typing import Callable

from .providers import Provider

_PROPOSE = ("Open the discussion: state a concrete position or answer on the topic, "
            "with brief reasoning.")
_RESPOND = ("Your turn. Respond on the merits to the last message. If you have no "
            "substantive objection left, begin your reply with AGREE and stop.")
_CRITIQUE = ("Your turn. Critique the last message strictly on the merits. If no "
             "substantive objection remains, begin your reply with AGREE and close.")
_SECRETARY = (
    "You are an impartial secretary of a debate, NOT a participant. From the "
    "transcript on the topic «{q}», produce a tight summary in EXACTLY this format:\n\n"
    "Points of agreement:\n- ...\n"
    "Residual disagreements:\n- ... (or 'none')\n"
    "Consensus / recommendation: ...\n\n"
    "Use ONLY what was said in the transcript. Add nothing of your own."
)

OnTurn = Callable[[str, str], None] | None


def _render(transcript: list) -> str:
    return "\n\n".join(f"{t['who']}: {t['text']}" for t in transcript) or "(empty)"


def _agreed(text: str) -> bool:
    """Did this turn concede? (reply opens with AGREE)."""
    head = text.strip().lower()[:24]
    return head.startswith(("agree", "agreed"))


class Council:
    def __init__(self, provider_a: Provider, provider_b: Provider, *,
                 name_a: str | None = None, name_b: str | None = None,
                 system: str | None = None):
        self.a, self.b = provider_a, provider_b
        self.name_a = name_a or getattr(provider_a, "name", "A")
        self.name_b = name_b or getattr(provider_b, "name", "B")
        # disambiguate equal names (same provider in both seats)
        if self.name_a == self.name_b:
            self.name_a, self.name_b = f"{self.name_a}-1", f"{self.name_b}-2"
        self.system = system or (
            "You are a rigorous debater. Argue on the merits, be concise, and concede "
            "plainly when you are wrong — do not argue for the sake of it.")

    async def _say(self, provider: Provider, question: str, transcript: list,
                   instruction: str) -> str:
        prompt = (f"Topic: {question}\n\nDiscussion so far:\n{_render(transcript)}\n\n"
                  f"{instruction}")
        return await provider.chat(self.system, [], prompt)

    async def debate(self, question: str, rounds: int = 2,
                     on_turn: OnTurn = None) -> list:
        """Run up to `rounds` exchanges (A then B). Stops early if B concedes."""
        transcript: list = []
        for _ in range(rounds):
            a = await self._say(self.a, question, transcript,
                                 _PROPOSE if not transcript else _RESPOND)
            transcript.append({"who": self.name_a, "text": a})
            if on_turn:
                on_turn(self.name_a, a)
            b = await self._say(self.b, question, transcript, _CRITIQUE)
            transcript.append({"who": self.name_b, "text": b})
            if on_turn:
                on_turn(self.name_b, b)
            if _agreed(b):
                break
        return transcript

    async def consensus(self, question: str, rounds: int = 2, on_turn: OnTurn = None,
                        secretary: Provider | None = None) -> dict:
        """Debate, then an impartial pass distills the consensus.
        Returns {'transcript': [...], 'verdict': str}."""
        transcript = await self.debate(question, rounds, on_turn)
        sec = secretary or self.a
        verdict = await sec.chat(_SECRETARY.format(q=question), [],
                                 "Transcript:\n" + _render(transcript))
        return {"transcript": transcript, "verdict": verdict}
