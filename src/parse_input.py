from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ParameterSchema(BaseModel):
    """Schema for a single function parameter."""

    type: Literal["string", "number", "boolean"]


class FunctionDefinition(BaseModel):
    """Validated function definition from functions_definition.json."""

    name: str = Field(..., min_length=1)
    description: str
    parameters: dict[str, ParameterSchema]
    returns: dict[str, Any]


class InputPrompt(BaseModel):
    """Validated prompt item from function_calling_tests.json."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only prompts early."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v
