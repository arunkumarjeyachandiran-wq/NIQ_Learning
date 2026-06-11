import csv
import json
from pathlib import Path
from typing import Iterable

from Agents.Models import AssetsOutput, ChartSpec, ImageSpec, TableSpec

ASSETS_DIR = Path("generated_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(label: str, suffix: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label
    ).strip("_")
    return f"{sanitized[:60]}.{suffix}"


def _write_table_file(table: TableSpec, index: int) -> str:
    path = ASSETS_DIR / _safe_filename(f"table_{index}_{table.title}", "csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(table.headers)
        writer.writerows(table.rows)
    return str(path)


def _write_chart_file(chart: ChartSpec, index: int) -> str:
    path = ASSETS_DIR / _safe_filename(f"chart_{index}_{chart.title}", "json")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(chart.model_dump(), stream, indent=2)
    return str(path)


def _write_image_placeholder(image: ImageSpec, index: int) -> str:
    path = ASSETS_DIR / _safe_filename(f"image_{index}_{image.title}", "json")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(image.model_dump(), stream, indent=2)
    return str(path)


def build_assets(
    tables: Iterable[TableSpec],
    charts: Iterable[ChartSpec],
    images: Iterable[ImageSpec],
) -> AssetsOutput:
    """Validate and persist asset specifications to the asset directory."""
    for index, table in enumerate(tables, start=1):
        _write_table_file(table, index)

    for index, chart in enumerate(charts, start=1):
        _write_chart_file(chart, index)

    for index, image in enumerate(images, start=1):
        _write_image_placeholder(image, index)

    return AssetsOutput(
        tables=list(tables),
        charts=list(charts),
        images=list(images),
    )
