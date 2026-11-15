"""The command `python -m mesqual.skills install` and the shipped skill `mesqual-study`."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import mesqual.skills
from mesqual.skills import _main
from mesqual.testing import make_dummy_study
from tests.helpers import SKILL, run_python_blocks

SKILL_FOLDER = Path('.agents', 'skills', 'mesqual-study')
NAME = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


def front_matter_and_body(text: str) -> tuple[dict[str, str], list[str]]:
    """The `key: value` lines of the YAML front matter, and the lines of the body."""
    lines = text.splitlines()
    assert lines[0] == '---'
    end = lines.index('---', 1)
    return dict(line.split(': ', 1) for line in lines[1:end]), lines[end + 1 :]


def install(project: Path, capsys) -> list[str]:
    _main(['install', str(project)])
    return capsys.readouterr().out.splitlines()


class TestInstall:
    def test_creates_the_skill_in_the_agents_folder(self, tmp_path, capsys):
        assert install(tmp_path, capsys) == [f'{tmp_path / SKILL_FOLDER}: created']
        assert (tmp_path / SKILL_FOLDER / 'SKILL.md').read_bytes() == SKILL.read_bytes()
        assert not (tmp_path / '.claude').exists()

    def test_a_second_install_changes_nothing(self, tmp_path, capsys):
        install(tmp_path, capsys)
        assert install(tmp_path, capsys) == [f'{tmp_path / SKILL_FOLDER}: unchanged']

    def test_overwrites_the_skill_file_and_keeps_other_files(self, tmp_path, capsys):
        install(tmp_path, capsys)
        (tmp_path / SKILL_FOLDER / 'SKILL.md').write_text('old version', encoding='utf-8')
        notes = tmp_path / SKILL_FOLDER / 'notes.md'
        notes.write_text('own notes', encoding='utf-8')
        assert install(tmp_path, capsys) == [f'{tmp_path / SKILL_FOLDER}: updated']
        assert (tmp_path / SKILL_FOLDER / 'SKILL.md').read_bytes() == SKILL.read_bytes()
        assert notes.read_text(encoding='utf-8') == 'own notes'

    @pytest.mark.parametrize('project', ['missing', 'file.txt'])
    def test_a_project_that_is_no_directory_exits_with_a_message(self, tmp_path, capsys, project):
        (tmp_path / 'file.txt').write_text('', encoding='utf-8')
        with pytest.raises(SystemExit) as exit_info:
            _main(['install', str(tmp_path / project)])
        assert exit_info.value.code == 2
        assert f'{tmp_path / project} is not an existing directory' in capsys.readouterr().err

    def test_no_shipped_skill_exits_with_a_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mesqual.skills, 'files', lambda package: tmp_path / 'empty')
        (tmp_path / 'empty').mkdir()
        with pytest.raises(SystemExit) as exit_info:
            _main(['install', str(tmp_path)])
        assert exit_info.value.code == 'no shipped skills found'  # Python prints it to stderr and exits with 1

    def test_the_module_runs_as_a_command_and_claude_adds_the_claude_folder(self, tmp_path):
        command = [sys.executable, '-m', 'mesqual.skills', 'install', str(tmp_path), '--claude']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        claude_folder = Path('.claude', 'skills', 'mesqual-study')
        assert result.stdout.splitlines() == [
            f'{tmp_path / SKILL_FOLDER}: created',
            f'{tmp_path / claude_folder}: created',
        ]
        for folder in (SKILL_FOLDER, claude_folder):
            assert (tmp_path / folder / 'SKILL.md').read_bytes() == SKILL.read_bytes()


class TestSkill:
    def test_the_front_matter_follows_the_agent_skills_format(self):
        fields, body = front_matter_and_body(SKILL.read_text(encoding='utf-8'))
        assert set(fields) == {'name', 'description'}
        assert NAME.match(fields['name']) and len(fields['name']) <= 64
        assert fields['name'] == SKILL.parent.name
        assert 0 < len(fields['description']) <= 1024
        assert ': ' not in fields['description']  # a plain YAML scalar
        assert len(body) < 500

    def test_the_front_matter_is_yaml_with_the_same_fields(self):
        yaml = pytest.importorskip('yaml')
        text = SKILL.read_text(encoding='utf-8')
        assert yaml.safe_load(text.split('---')[1]) == front_matter_and_body(text)[0]

    def test_the_python_blocks_run_in_order_on_the_synthetic_study(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_python_blocks(SKILL, {'__name__': 'skill_examples', 'study': make_dummy_study()})
        assert (tmp_path / 'price_map.html').is_file() and (tmp_path / 'prices.html').is_file()
