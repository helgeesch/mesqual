import subprocess
import sys

HIDE_SELENIUM = "import sys; sys.modules['selenium'] = None; "


def _run_without_selenium(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', HIDE_SELENIUM + code], capture_output=True, text=True)


def test_folium_utils_import_without_selenium():
    result = _run_without_selenium('import mesqual.utils.folium_utils')
    assert result.returncode == 0, result.stderr


def test_screenshotter_without_selenium_names_the_extra():
    result = _run_without_selenium(
        'from mesqual.utils.folium_utils.screenshotter import FoliumScreenshotter; FoliumScreenshotter("map.html")'
    )
    assert result.returncode != 0
    assert 'ImportError' in result.stderr and 'mesqual[screenshot]' in result.stderr
