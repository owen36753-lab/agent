"""Conservative object grounding helpers for ALFWorld actions.

The grounding rule is directional:

- exact object names match;
- a candidate subclass may satisfy a superclass goal;
- sibling categories do not satisfy each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SUPERCLASS_MEMBERS: dict[str, set[str]] = {
    "bottle": {"bottle", "spraybottle", "soapbottle", "waterbottle"},
    "cloth": {"cloth", "dishcloth", "handtowel", "rag", "towel", "washcloth"},
    "cookware": {"cookware", "pan", "pot"},
    "cup": {"cup", "mug"},
}


OBJECT_ACTION_RE = re.compile(
    r"^(?:take|move|heat|cool|clean|slice)\s+([a-z]+)\s+(\d+)\b"
)


@dataclass(frozen=True)
class GroundingDecision:
    goal_object: str
    candidate_object: str
    matches: bool
    relation: str
    reason: str


def normalize_object_name(name: str | None) -> str:
    """Normalize ALFWorld-ish object names without aggressive stemming."""
    if not name:
        return ""
    normalized = re.sub(r"[^a-z]", "", name.lower())
    if normalized.endswith("ies") and len(normalized) > 4:
        return f"{normalized[:-3]}y"
    if normalized.endswith("s") and not normalized.endswith(("ss", "basin")):
        singular = normalized[:-1]
        if singular in SUPERCLASS_MEMBERS or any(
            singular in members for members in SUPERCLASS_MEMBERS.values()
        ):
            return singular
    return normalized


def object_base_from_instance(object_instance: str | None) -> str:
    if not object_instance:
        return ""
    match = re.match(r"\s*([a-z]+)\s+\d+\s*$", object_instance.lower())
    return normalize_object_name(match.group(1) if match else object_instance)


def object_instance_from_action(action: str | None) -> str | None:
    if not action:
        return None
    match = OBJECT_ACTION_RE.search(action.strip().lower())
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)}"


def is_candidate_for_goal(goal_object: str | None, candidate_object: str | None) -> GroundingDecision:
    goal = normalize_object_name(goal_object)
    candidate = normalize_object_name(candidate_object)
    if not goal or not candidate:
        return GroundingDecision(goal, candidate, False, "unknown", "Missing goal or candidate object.")

    if goal == candidate:
        return GroundingDecision(goal, candidate, True, "exact", "Candidate exactly matches goal object.")

    members = SUPERCLASS_MEMBERS.get(goal, set())
    if candidate in members:
        return GroundingDecision(
            goal,
            candidate,
            True,
            "subclass",
            f"Candidate {candidate!r} is treated as a subtype of goal class {goal!r}.",
        )

    for superclass, superclass_members in SUPERCLASS_MEMBERS.items():
        if goal in superclass_members and candidate in superclass_members:
            return GroundingDecision(
                goal,
                candidate,
                False,
                "sibling",
                (
                    f"Candidate {candidate!r} and goal {goal!r} are both under "
                    f"{superclass!r}, but sibling categories are not interchangeable."
                ),
            )

    return GroundingDecision(
        goal,
        candidate,
        False,
        "unknown",
        f"No conservative is-a relation from candidate {candidate!r} to goal {goal!r}.",
    )


def action_object_matches_goal(goal_object: str | None, action: str | None) -> GroundingDecision | None:
    object_instance = object_instance_from_action(action)
    if not object_instance:
        return None
    return is_candidate_for_goal(goal_object, object_base_from_instance(object_instance))


def build_object_grounding(
    goal_object: str | None,
    step_logs: list[dict[str, Any]],
    admissible_commands: list[str],
    treatment: str | None = None,
) -> dict[str, Any]:
    """Summarize currently grounded objects for a parsed ALFWorld task."""
    normalized_goal = normalize_object_name(goal_object)
    current_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []

    for action in admissible_commands:
        object_instance = object_instance_from_action(action)
        if not object_instance:
            continue
        base = object_base_from_instance(object_instance)
        decision = is_candidate_for_goal(normalized_goal, base)
        record = {
            "object_instance": object_instance,
            "object_base": base,
            "action": action,
            "relation": decision.relation,
            "reason": decision.reason,
        }
        if decision.matches:
            current_candidates.append(record)
        elif decision.relation == "sibling":
            rejected_candidates.append(record)

    held_object: str | None = None
    held_object_base: str | None = None
    treated_objects: set[str] = set()
    grounded_history: list[dict[str, Any]] = []

    for step in step_logs:
        action = str(step.get("action", "")).strip().lower()
        object_instance = object_instance_from_action(action)
        if not object_instance:
            continue
        base = object_base_from_instance(object_instance)
        decision = is_candidate_for_goal(normalized_goal, base)
        if not decision.matches:
            continue

        grounded_history.append(
            {
                "step": step.get("step"),
                "action": step.get("action"),
                "object_instance": object_instance,
                "object_base": base,
                "relation": decision.relation,
            }
        )
        if action.startswith(f"take {object_instance} from "):
            held_object = object_instance
            held_object_base = base
        if treatment and action.startswith(f"{treatment} {object_instance} with "):
            treated_objects.add(object_instance)
            held_object = object_instance
            held_object_base = base
        if action.startswith(f"move {object_instance} to "):
            held_object = None
            held_object_base = None

    return {
        "goal_object": normalized_goal,
        "current_candidates": current_candidates,
        "rejected_candidates": rejected_candidates,
        "held_object": held_object,
        "held_object_base": held_object_base,
        "treated_objects": sorted(treated_objects),
        "grounded_history": grounded_history,
    }
