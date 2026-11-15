"""Agent skills of mesqual: `python -m mesqual.skills install <project>`, see https://docs.mesqual.io/use-with-agents/."""

import argparse
import sys
from importlib.resources import files
from pathlib import Path


def _write_if_changed(content: bytes, target: Path) -> str:
    """Write `content` to `target` and return `created`, `updated` or `unchanged`."""
    if target.is_file() and target.read_bytes() == content:
        return 'unchanged'
    status = 'updated' if target.is_file() else 'created'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return status


def _main(argv: list[str] | None = None) -> None:
    """Copy the `SKILL.md` of each shipped skill into the project and print each skill folder with its status.

    Invalid arguments exit with code 2, and no shipped skill with code 1, each with a message on stderr.
    """
    parser = argparse.ArgumentParser(prog='python -m mesqual.skills', description='Install the skills of mesqual.')
    commands = parser.add_subparsers(dest='command', required=True)
    install = commands.add_parser('install', help='Copy the shipped skills into <project>/.agents/skills/.')
    install.add_argument('project', type=Path, help='The project folder.')
    install.add_argument('--claude', action='store_true', help='Also copy the skills into <project>/.claude/skills/.')
    args = parser.parse_args(argv)
    if not args.project.is_dir():
        install.error(f'{args.project} is not an existing directory')
    targets = [args.project / '.agents' / 'skills'] + ([args.project / '.claude' / 'skills'] if args.claude else [])
    skills = sorted((f for f in files(__name__).iterdir() if (f / 'SKILL.md').is_file()), key=lambda f: f.name)
    if not skills:
        sys.exit('no shipped skills found')
    for target in targets:
        for skill in skills:
            status = _write_if_changed((skill / 'SKILL.md').read_bytes(), target / skill.name / 'SKILL.md')
            print(f'{target / skill.name}: {status}')
