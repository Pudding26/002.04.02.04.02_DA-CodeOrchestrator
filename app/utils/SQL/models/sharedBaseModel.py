from pydantic import BaseModel
from typing import Any
import json

class SharedBaseModel(BaseModel):
    """
    Shared base for all Pydantic models in the application.
    """
    class Config:
        orm_mode = True
        anystr_strip_whitespace = True
        use_enum_values = True
        extra = "forbid"

    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert to dict."""
        return self.model_dump(exclude_none=exclude_none)

    def json_api(self, **kwargs: Any) -> str:
        """Pretty JSON for APIs/logs/debugging."""
        return self.model_dump_json(indent=2, exclude_none=True, **kwargs)
