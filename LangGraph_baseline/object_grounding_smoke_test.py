"""Smoke tests for conservative object grounding decisions."""

from __future__ import annotations

from object_grounding import build_object_grounding, is_candidate_for_goal


CASES = [
    ("cloth", "handtowel", True, "subclass"),
    ("cloth", "towel", True, "subclass"),
    ("cloth", "apple", False, "unknown"),
    ("bottle", "spraybottle", True, "subclass"),
    ("bottle", "soapbottle", True, "subclass"),
    ("spraybottle", "soapbottle", False, "sibling"),
    ("pan", "pot", False, "sibling"),
    ("pot", "pan", False, "sibling"),
    ("cookware", "pan", True, "subclass"),
    ("cookware", "pot", True, "subclass"),
    ("cup", "mug", True, "subclass"),
    ("mug", "cup", False, "sibling"),
    ("apple", "apple", True, "exact"),
    ("apple", "potato", False, "unknown"),
]


def main() -> None:
    print("goal,candidate,expected,actual,relation")
    for goal, candidate, expected, expected_relation in CASES:
        decision = is_candidate_for_goal(goal, candidate)
        print(
            f"{goal},{candidate},{expected},{decision.matches},{decision.relation}"
        )
        assert decision.matches is expected
        assert decision.relation == expected_relation

    grounding = build_object_grounding(
        "cloth",
        step_logs=[
            {
                "step": 1,
                "action": "take handtowel 1 from handtowelholder 1",
            }
        ],
        admissible_commands=[
            "clean handtowel 1 with sinkbasin 1",
            "move handtowel 1 to cabinet 1",
            "take apple 1 from countertop 1",
        ],
        treatment="clean",
    )
    assert grounding["held_object"] == "handtowel 1"
    assert grounding["held_object_base"] == "handtowel"
    assert grounding["current_candidates"][0]["action"] == "clean handtowel 1 with sinkbasin 1"
    assert len(grounding["current_candidates"]) == 2
    print("grounding_summary_ok=True")


if __name__ == "__main__":
    main()
