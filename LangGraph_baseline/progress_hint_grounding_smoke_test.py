"""Smoke test that graph progress hints use object grounding."""

from __future__ import annotations

from graph_main import _build_progress_hint
from object_grounding import build_object_grounding


def main() -> None:
    goal = "clean some cloth and put it in cabinet"
    actions = [
        "take handtowel 1 from handtowelholder 1",
        "go to sinkbasin 1",
        "take apple 1 from countertop 1",
    ]
    grounding = build_object_grounding("cloth", [], actions, "clean")
    hint = _build_progress_hint(goal, [], actions, grounding)
    print(hint)
    assert hint is not None
    assert "take handtowel 1 from handtowelholder 1" in hint

    history = [
        {
            "step": 1,
            "action": "take handtowel 1 from handtowelholder 1",
        }
    ]
    actions = [
        "clean handtowel 1 with sinkbasin 1",
        "move handtowel 1 to cabinet 1",
    ]
    grounding = build_object_grounding("cloth", history, actions, "clean")
    hint = _build_progress_hint(goal, history, actions, grounding)
    print(hint)
    assert hint is not None
    assert "clean handtowel 1 with sinkbasin 1" in hint


if __name__ == "__main__":
    main()
