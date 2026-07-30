from pydantic import BaseModel

class ParsePrompt(BaseModel):
    prompt : str

class ParseFunc(BaseModel):
    name: str
    description: str
    parameters: dict
    returns: dict