# app/routers/idea_router.py
import logging
from fastapi import APIRouter, HTTPException, status, Body
from google.api_core.exceptions import GoogleAPIError # For specific Google errors

from app.models.idea_models import IdeaInput, ScopeOutput, ErrorResponse
from app.services.gemini_service import gemini_service # Import the Gemini service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/generate-scope",
    response_model=ScopeOutput,
    summary="Generate MVP Scope Document (using Google Gemini AI)",
    description=(
        "Accepts a startup idea and its target industry, then utilizes Google's Gemini AI "
        "to generate a comprehensive MVP (Minimum Viable Product) scope document. "
        "The document includes sections like Executive Summary, Core Features, Tech Stack, and more."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Successfully generated MVP scope document."},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Invalid input data or content policy violation by AI."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse, "description": "Validation error in the request body."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "An unexpected internal server error occurred."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "AI service (Gemini) is unavailable or encountered an error (e.g., quota, permission)."}
    },
    tags=["AI Scope Generation"] # Group related endpoints in Swagger UI
)
async def generate_scope_endpoint(
    idea_input: IdeaInput = Body(
        ...,
        examples={ # Adding examples for the /docs page
            "default": {
                "summary": "Default Example",
                "description": "A typical valid request.",
                "value": {
                    "idea": "A mobile application that connects local artisans with buyers interested in unique, handmade crafts.",
                    "industry": "E-commerce and Local Artisan Marketplace"
                }
            },
            "fitness_app": {
                "summary": "Fitness App Example",
                "value": {
                    "idea": "An AI-powered fitness coaching app that creates personalized workout and meal plans based on user goals and biometrics.",
                    "industry": "Health & Wellness Technology"
                }
            }
        }
    )
):
    """
    Endpoint to generate an MVP scope document for a given startup idea and industry.
    """
    logger.info(f"IdeaRouter: Received request for '/generate-scope'. Industry='{idea_input.industry}', Idea='{idea_input.idea[:70]}...'")
    try:
        generated_text = await gemini_service.generate_startup_scope(idea_input)
        # Basic check, though service should also ensure non-empty
        if not generated_text or not generated_text.strip():
            logger.error("IdeaRouter: Gemini service returned empty or whitespace-only content.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI service (Gemini) failed to generate meaningful content."
            )
        return ScopeOutput(generated_scope=generated_text)

    except ConnectionError as e: # Custom error from our service for config/API issues
        logger.error(f"IdeaRouter: ConnectionError from GeminiService. Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    except ValueError as e: # Raised by GeminiService for content blocking or invalid internal state
        logger.warning(f"IdeaRouter: ValueError from GeminiService (e.g., content policy). Error: {e}", exc_info=False)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # No need to explicitly catch GoogleAPIError here if GeminiService handles and re-raises as ConnectionError
    # or if specific GoogleAPIError types are handled there for more context.

    except Exception as e: # Catch-all for truly unexpected errors in the router
        logger.critical(f"IdeaRouter: Unexpected critical error in '/generate-scope' endpoint. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred. The technical team has been notified."
        )