from pydantic import BaseModel, ValidationError
from typing import Dict, Any

class FunctionParameter(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, str]

class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]

def validate_output(data: dict) -> tuple[bool, str | None]:
    try:
        FunctionCallOutput(**data)
        return True, None
    except ValidationError as e:
        return False, str(e)