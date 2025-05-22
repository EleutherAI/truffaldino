from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters

class OpenAIFunctionToolSchema(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition 