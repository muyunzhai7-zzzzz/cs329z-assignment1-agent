"""Check tool calls before execution and return errors suitable for model feedback.

Q6 parses the model's raw text; this registry checks the parsed name and args.
The Q7 loop must send each ToolResult.content back to the model unchanged so
that it can use validation errors to correct its next call.
"""

from __future__ import annotations

from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    """Expose tool schemas, validate without side effects, then execute once."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._validators = {
            name: Draft202012Validator(tool.parameters)
            for name, tool in self._tools.items()
        }

    def schemas(self) -> list[dict]:
        """Only model-facing metadata, in registration order."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.parameters),
            }
            for tool in self._tools.values()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """Reject bad calls without running the tool or guessing corrections."""
        if not isinstance(name, str) or name not in self._tools:
            available = ", ".join(self._tools) or "(none registered)"
            return ToolResult(
                "unknown_tool",
                f"Unknown tool {name!r}. Available tools: {available}. "
                "Choose an exact registered tool name.",
            )
        if not isinstance(args, dict):
            return ToolResult(
                "invalid_args",
                f"Arguments for {name!r} must be a JSON object mapping parameter "
                "names to values. Correct the arguments and try again.",
            )
        try:
            self._validators[name].validate(args)
        except ValidationError as exc:
            location = "$" + "".join(f"[{part!r}]" for part in exc.absolute_path)
            return ToolResult(
                "invalid_args",
                f"Invalid arguments for {name!r} at {location}: {exc.message}. "
                "Correct the arguments and try again.",
            )
        except Exception as exc:
            return ToolResult(
                "invalid_args",
                f"Argument validation failed for {name!r}: {type(exc).__name__}: {exc}",
            )
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        """Every execution passes validation; errors do not stop the registry."""
        error = self.validate(name, args)
        if error is not None:
            return error
        try:
            value = self._tools[name].fn(**args)
            return ToolResult("ok", str(value))
        except Exception as exc:
            # Do not retry automatically: a tool may already have side effects.
            return ToolResult(
                "tool_error",
                f"Tool {name!r} failed: {type(exc).__name__}: {exc}",
            )
