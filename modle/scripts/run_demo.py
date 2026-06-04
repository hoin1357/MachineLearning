from __future__ import annotations

import os
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def main() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("APP_PORT", "8765"))
    browser_url = f"http://127.0.0.1:{port}"

    if os.getenv("OPEN_BROWSER", "1") != "0":
        def open_browser() -> None:
            deadline = time.time() + 60
            server_ready = False
            while time.time() < deadline:
                try:
                    with urlopen(browser_url, timeout=1):
                        server_ready = True
                        break
                except (OSError, URLError):
                    time.sleep(0.5)
            if server_ready:
                webbrowser.open(browser_url)
            else:
                print(f"Server did not respond within 60 seconds: {browser_url}")

        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting local server: {browser_url}")
    print("Press Ctrl+C to stop.")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
