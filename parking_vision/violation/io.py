import json

from perception_types import ensure_parent_dir

from .types import PipelineFrameResult


def save_jsonl(output_path, items: list[PipelineFrameResult]) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
