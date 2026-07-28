from pydantic import BaseModel
from typing import Dict, Optional, Any

class ParameterDef(BaseModel):
    type: str
    description: Optional[str] = None

class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, ParameterDef]
    returns: Dict[str, str]

class OutputItem(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]