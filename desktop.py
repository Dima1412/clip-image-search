"""
Запускает Streamlit в фоне и открывает его в Edge/Chrome в app-режиме.
Запуск: start.bat  |  Остановить: Ctrl+C
"""

import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent

_EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, timeout: int = 60) -> bool:
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _find_browser() -> str | None:
    for path in _EDGE_PATHS + _CHROME_PATHS:
        if Path(path).exists():
            return path
    return None


def main() -> None:
    port = _free_port()

    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    python = str(venv_python) if venv_python.exists() else sys.executable

    print(f"Запускаю на порту {port}...", flush=True)

    streamlit_proc = subprocess.Popen(
        [
            python, "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
    )

    try:
        if not _wait_ready(port):
            streamlit_proc.kill()
            sys.exit("Streamlit не запустился за 60 секунд")

        browser = _find_browser()
        url = f"http://localhost:{port}"

        if browser:
            user_data_dir = tempfile.mkdtemp(prefix="clip-search-")
            try:
                browser_proc = subprocess.Popen([
                    browser,
                    f"--app={url}",
                    f"--user-data-dir={user_data_dir}",
                    "--window-size=1280,820",
                ])
                browser_proc.wait()
            finally:
                shutil.rmtree(user_data_dir, ignore_errors=True)
        else:
            import webbrowser
            webbrowser.open(url)
            streamlit_proc.wait()

    except KeyboardInterrupt:
        pass
    finally:
        streamlit_proc.terminate()
        streamlit_proc.wait()


if __name__ == "__main__":
    main()
