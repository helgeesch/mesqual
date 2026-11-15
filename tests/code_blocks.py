"""Fenced code blocks of Markdown files, for the tests that run the README, the architecture example and the guides."""

import re

FENCE = re.compile(r'^([ \t]*)(```|~~~)(.*)$', re.MULTILINE)
CODE_BLOCK = re.compile(r'^```(\w*)\n(.*?)^```$', re.MULTILINE | re.DOTALL)


def code_blocks(text: str) -> list[tuple[str, str]]:
    """All fenced code blocks of a Markdown text as `(language tag, body)`, in order."""
    return CODE_BLOCK.findall(text)


def fence_tags(text: str) -> set[str]:
    """The language tags of the opening fences of a Markdown text.

    Asserts the fence shape: the text has fences, each unindented and of
    backticks, and no closing fence has a tag.
    """
    fences = FENCE.findall(text)
    assert fences, 'no fenced code blocks'
    assert all(indent == '' and mark == '```' for indent, mark, _ in fences), 'an indented or ~~~ fence'
    assert {tag for _, _, tag in fences[1::2]} == {''}, 'a closing fence with a tag'
    return {tag for _, _, tag in fences[::2]}


def python_blocks(text: str) -> list[str]:
    """The `python` code blocks of a Markdown text, in order."""
    return [body for tag, body in code_blocks(text) if tag == 'python']
