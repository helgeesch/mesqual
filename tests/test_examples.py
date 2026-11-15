import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).parents[1] / 'examples'


def test_quickstart_runs_and_saves_the_dashboard(tmp_path):
    spec = importlib.util.spec_from_file_location('quickstart', EXAMPLES / 'quickstart.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_file = tmp_path / 'quickstart.html'
    quickstart = module.Quickstart(output_file)
    quickstart.describe()
    quickstart.run()
    assert output_file.stat().st_size > 0
