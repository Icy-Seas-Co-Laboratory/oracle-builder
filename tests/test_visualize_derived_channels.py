from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from oracle_builder.data.sqlite_dataset import create_synthetic_classification


def test_derived_channel_visualizer_writes_native_size_three_column_sheet(tmp_path):
    database = tmp_path / "classification.sqlite"
    output = tmp_path / "derived_channels.png"
    create_synthetic_classification(database, n=4, shape=(10, 16, 1), classes=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_derived_channels.py",
            "--database",
            str(database),
            "--output",
            str(output),
            "--count",
            "2",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    with Image.open(output) as sheet:
        assert sheet.width == 3 * 16 + 4 * 12
        assert sheet.height > 2 * 10
    assert "2 ROI rows × 3 channels" in result.stdout
