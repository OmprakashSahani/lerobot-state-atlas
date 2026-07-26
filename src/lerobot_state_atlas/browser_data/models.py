"""Small immutable result models for browser-data operations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BrowserDataExport:
    """Result of a successful atomic browser-data export."""

    output_path: Path
    bundle_id: str
    dataset_frame_count: int
    tool_point_visit_count: int
    arm_voxel_entry_count: int
    unique_shared_grid_cell_count: int
    payload_byte_count: int
