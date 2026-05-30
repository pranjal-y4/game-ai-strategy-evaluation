from __future__ import annotations
import csv
import os
import time


class TrainingLogger:


    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._f.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass
