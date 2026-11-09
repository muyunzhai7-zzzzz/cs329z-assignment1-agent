"""Q14: host-enforced command rules and an auditable source-origin signal."""

from __future__ import annotations

import json
import re

from cs329z_hw1.types import ToolCall, ToolResult


def _strings(value):
    """Inspect decoded source text too, rather than JSON escape sequences."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


class Guardrails:
    def __init__(self, deny_rules: tuple[str, ...]) -> None:
        # Snapshot trusted configuration once. Source text and model replies
        # have no interface for replacing these compiled rules.
        self._deny_rules = tuple(re.compile(rule) for rule in deny_rules)
        self._sources: list[tuple[str, int, str]] = []

    def check(self, call: ToolCall) -> ToolResult | None:
        if call.name != "run_terminal":
            return None
        command = call.args.get("cmd")
        if not isinstance(command, str):
            return None  # The registry is responsible for schema validation.
        for rule in self._deny_rules:
            if rule.search(command):
                return ToolResult(
                    "denied",
                    f"Command denied by configured rule {rule.pattern!r}. Nothing was executed. "
                    "Approval cannot override this rule. Do not disguise the same prohibited "
                    "operation; continue with permitted work or explain the limitation.",
                )
        return None

    def observe(self, name: str, result: ToolResult, event_index: int) -> None:
        # ask_user is an actual human answer; remember echoes a fact supplied
        # for storage. Neither is a retrieved external instruction source.
        if name in ("ask_user", "remember") or result.status not in ("ok", "tool_error"):
            return
        self._sources.append((name, event_index, result.content))
        try:
            decoded = json.loads(result.content)
        except (ValueError, TypeError, RecursionError):
            return
        self._sources.extend((name, event_index, text) for text in _strings(decoded))

    def source_warning(self, call: ToolCall) -> dict | None:
        if call.name != "run_terminal":
            return None
        command = call.args.get("cmd")
        if not isinstance(command, str) or not command.strip():
            return None
        for name, event_index, text in self._sources:
            if command in text:
                # Exact overlap is evidence of possible copying, not proof of
                # malicious intent. It must not automatically reject a setup
                # guide the user explicitly asked the agent to follow.
                return {
                    "type": "source_warning", "name": call.name, "cmd": command,
                    "source_tool": name, "source_event": event_index,
                    "content": (
                        "The proposed command appears verbatim in an earlier tool observation. "
                        "A source match is not authorization or proof of an attack. "
                        "Configured deny rules and the normal approval policy still apply."
                    ),
                }
        return None
