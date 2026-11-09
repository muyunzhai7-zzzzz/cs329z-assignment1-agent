"""Q10: execute through the provided runner and show bounded observations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cs329z_hw1 import sandbox
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_tokens
from cs329z_hw1.types import AgentConfig


class TerminalExecutionError(RuntimeError):
    """The registry converts this readable observation to a tool_error."""


def _line_count(text: str) -> int:
    return text.count("\n") + int(bool(text) and not text.endswith("\n"))


def _preview(text: str, characters: int) -> str:
    if len(text) <= characters:
        return text
    notice = "\n... [preview shortened; full captured output saved below] ...\n"
    if characters <= len(notice):
        return text[:characters]
    available = characters - len(notice)
    head = available * 2 // 3
    tail = available - head
    return text[:head] + notice + (text[-tail:] if tail else "")


class TerminalTool:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Keep room for the system prompt, commands, and several observations.
        # Full-session compaction is implemented separately in Q11.
        self.preview_tokens = min(600, max(100, config.context_budget // 10))

    def _save_output(self, stdout: str, stderr: str) -> tuple[str, str]:
        workspace = (
            Path(self.config.workspace).resolve()
            if self.config.workspace is not None else sandbox.default_workspace()
        )
        directory = workspace / ".terminal_outputs"
        if not directory.resolve().is_relative_to(workspace):
            raise ValueError("The output directory must stay inside the workspace.")
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((stdout + "\0" + stderr).encode("utf-8")).hexdigest()[:24]
        paths = []
        for stream, text in (("stdout", stdout), ("stderr", stderr)):
            path = directory / f"{digest}.{stream}.txt"
            # Exclusive creation avoids following an existing file or symlink.
            try:
                with path.open("x", encoding="utf-8", newline="") as output:
                    output.write(text)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != text.encode("utf-8"):
                    raise ValueError("An output path already exists with different content.")
            paths.append(path.relative_to(workspace).as_posix())
        return paths[0], paths[1]

    def _observation(self, result: sandbox.TerminalResult) -> str:
        header = (
            f"exit_code: {result.exit_code}\n"
            f"timed_out: {result.timed_out}\n"
            f"rejected: {result.rejected}\n"
            f"reason: {result.reason or '(none)'}\n"
            f"stdout_characters: {len(result.stdout)}; stdout_lines: {_line_count(result.stdout)}\n"
            f"stderr_characters: {len(result.stderr)}; stderr_lines: {_line_count(result.stderr)}\n"
        )
        complete = header + "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr
        if count_tokens(complete) <= self.preview_tokens:
            return complete

        stdout_path, stderr_path = self._save_output(result.stdout, result.stderr)
        footer = (
            "\nCaptured output files, relative to the command workspace:\n"
            f"stdout_file: {stdout_path}\nstderr_file: {stderr_path}\n"
            "Use a targeted command to inspect omitted content; sed -n 'Np' FILE "
            "reads physical line N (replace N with a number), counting empty lines.\n"
        )
        available = max(0, self.preview_tokens * CHARS_PER_TOKEN - len(header + footer + "stdout:\n\nstderr:\n"))
        stderr_space = min(len(result.stderr), available // 3)
        stdout_space = min(len(result.stdout), available - stderr_space)
        stderr_space = min(len(result.stderr), available - stdout_space)
        return (
            header + "stdout:\n" + _preview(result.stdout, stdout_space)
            + "\nstderr:\n" + _preview(result.stderr, stderr_space) + footer
        )

    def run(self, cmd: str) -> str:
        result = sandbox.run_terminal(
            cmd, workspace=self.config.workspace, timeout=self.config.terminal_timeout,
        )
        observation = self._observation(result)
        if result.rejected or result.timed_out or result.exit_code != 0:
            raise TerminalExecutionError(observation)
        return observation
