from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from schemas import ActionResult


@dataclass
class TrackActionState:
    stable_label: Optional[str] = None
    stable_score: float = 0.0
    candidate_label: Optional[str] = None
    candidate_score: float = 0.0
    candidate_count: int = 0


class ActionSmoother:
    def __init__(self, score_thr: float, confirm_count: int):
        self.score_thr = float(score_thr)
        self.confirm_count = max(int(confirm_count), 1)
        self.states: Dict[int, TrackActionState] = {}

    def update(self, track_id: int, raw: ActionResult) -> ActionResult:
        if raw.label is None:
            return raw

        state = self.states.setdefault(track_id, TrackActionState())

        if raw.score < self.score_thr:
            if state.stable_label is not None:
                return ActionResult(label=state.stable_label, score=state.stable_score, state="uncertain")
            return ActionResult(label=None, score=raw.score, state="uncertain")

        if state.stable_label is None:
            return self._promote_without_stable(state, raw)

        if raw.label == state.stable_label:
            state.stable_score = raw.score
            state.candidate_label = None
            state.candidate_score = 0.0
            state.candidate_count = 0
            return ActionResult(label=state.stable_label, score=raw.score, state="stable")

        if raw.label != state.candidate_label:
            state.candidate_label = raw.label
            state.candidate_score = raw.score
            state.candidate_count = 1
        else:
            state.candidate_count += 1
            state.candidate_score = raw.score

        if state.candidate_count >= self.confirm_count:
            state.stable_label = state.candidate_label
            state.stable_score = state.candidate_score
            state.candidate_label = None
            state.candidate_score = 0.0
            state.candidate_count = 0
            return ActionResult(label=state.stable_label, score=state.stable_score, state="stable")

        return ActionResult(label=state.stable_label, score=state.stable_score, state="transition")

    def prune(self, active_track_ids: Iterable[int]) -> None:
        active = set(int(track_id) for track_id in active_track_ids)
        stale = [track_id for track_id in self.states if track_id not in active]
        for track_id in stale:
            del self.states[track_id]

    def _promote_without_stable(self, state: TrackActionState, raw: ActionResult) -> ActionResult:
        if raw.label != state.candidate_label:
            state.candidate_label = raw.label
            state.candidate_score = raw.score
            state.candidate_count = 1
        else:
            state.candidate_count += 1
            state.candidate_score = raw.score

        if state.candidate_count >= self.confirm_count:
            state.stable_label = state.candidate_label
            state.stable_score = state.candidate_score
            state.candidate_label = None
            state.candidate_score = 0.0
            state.candidate_count = 0
            return ActionResult(label=state.stable_label, score=state.stable_score, state="stable")

        return ActionResult(label=None, score=raw.score, state="transition")
