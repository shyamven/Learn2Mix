from __future__ import annotations

import urllib.request

import nbformat
from nbclient import NotebookClient


def run_remote_notebook(url: str, timeout: int = 1200) -> int:
    with urllib.request.urlopen(url) as response:
        nb = nbformat.reads(response.read().decode("utf-8"), as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False)
    client.execute()
    return 0

