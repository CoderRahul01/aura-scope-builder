# app/models/idea_models.py
from pydantic import BaseModel, Field
from typing import Any # For more generic error responses if needed

class IdeaInput(BaseModel):
    """
    Input model for the idea generation endpoint.
    """
    idea: str = Field(
        ..., # Ellipsis means this field is required
        min_length=10,
        max_length=1000, # Added max_length for basic validation
        description="The core startup idea description (between 10 and 1000 characters)."
    )
    industry: str = Field(
        ...,
        min_length=3,
        max_length=100, # Added max_length
        description="The target industry for the idea (between 3 and 100 characters)."
    )

    # Example of how to add custom validation or examples
    # model_config = {
    #     "json_schema_extra": {
    #         "examples": [
    #             {
    #                 "idea": "A mobile app that uses AI to suggest personalized workout routines.",
    #                 "industry": "Health and Fitness Tech"
    #             }
    #         ]
    #     }
    # }


class ScopeOutput(BaseModel):
    """
    Output model containing the generated MVP scope.
    """
    generated_scope: str = Field(..., description="The AI-generated MVP scope document.")


class ErrorDetail(BaseModel):
    """
    Standard detail structure for validation errors or specific field errors.
    """
    loc: list[str | int] | None = None # Location of the error (e.g. ['body', 'idea'])
    msg: str
    type: str | None = None


class ErrorResponse(BaseModel):
    """
    Standard error response model for the API.
    """
    detail: str | list[ErrorDetail] # Can be a simple string or a list of detailed errors