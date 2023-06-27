from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    bound_sites: list[str]
    properties: list[str]
    path: Path
