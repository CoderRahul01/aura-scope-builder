# app/services/gemini_service.py
import logging
import asyncio # For running synchronous SDK calls in a thread pool
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, PermissionDenied

from app.core.config import settings
from app.models.idea_models import IdeaInput

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service class for interacting with the Google Gemini Pro API.
    Handles client configuration, prompt generation, API calls, and error handling.
    """
    def __init__(self):
        self.model = None
        if not settings.GEMINI_API_KEY:
            logger.error("GeminiService: GEMINI_API_KEY is not configured. Service will be non-operational.")
            return # Service remains non-operational if key is missing

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            logger.info(f"GeminiService initialized successfully with model: {settings.GEMINI_MODEL_NAME}.")
        except Exception as e:
            logger.error(f"GeminiService: Failed to configure or initialize GenerativeModel "
                         f"({settings.GEMINI_MODEL_NAME}). Error: {e}", exc_info=True)
            # self.model remains None, subsequent calls will fail gracefully.

    async def generate_startup_scope(self, idea_data: IdeaInput) -> str:
        """
        Generates an MVP scope document using Google Gemini.

        Args:
            idea_data: An IdeaInput Pydantic model containing the idea and industry.

        Returns:
            A string containing the generated MVP scope document.

        Raises:
            ConnectionError: If the service is not configured or a fatal API error occurs.
            ValueError: If content is blocked by safety settings or input is invalid.
        """
        if not self.model:
            logger.error("GeminiService.generate_startup_scope: Model not initialized due to configuration issues.")
            raise ConnectionError("Gemini AI service is not properly configured. Please check API key and server logs.")

        prompt = self._construct_prompt(idea_data)
        generation_config = self._get_generation_config()
        safety_settings = self._get_safety_settings()

        logger.info(f"GeminiService: Generating scope for industry='{idea_data.industry}', idea='{idea_data.idea[:70]}...'")

        try:
            # Crucial for FastAPI: Run synchronous SDK calls in a separate thread
            # to avoid blocking the main asyncio event loop.
            response = await asyncio.to_thread(
                self.model.generate_content, # The synchronous function
                prompt,                      # Positional arguments
                generation_config=generation_config, # Keyword arguments
                safety_settings=safety_settings,
                # request_options={"timeout": 60} # Optional: set a timeout for the API call
            )

            return self._process_gemini_response(response)

        except ResourceExhausted as e:
            logger.error(f"GeminiService: Quota exhausted. Error: {e}", exc_info=True)
            raise ConnectionError(f"Gemini API quota exceeded. Please check your Google Cloud project billing and limits. Details: {e.message}") from e
        except PermissionDenied as e:
            logger.error(f"GeminiService: Permission denied. Check API key and IAM permissions. Error: {e}", exc_info=True)
            raise ConnectionError(f"Gemini API permission denied. Ensure API key is valid and has Vertex AI permissions. Details: {e.message}") from e
        except GoogleAPIError as e:
            logger.error(f"GeminiService: A Google API error occurred. Error: {e}", exc_info=True)
            raise ConnectionError(f"An error occurred with the Gemini API: {e.message or str(e)}") from e
        except ValueError as e: # Catch our own ValueErrors (e.g., from _process_gemini_response)
            logger.warning(f"GeminiService: Value error during scope generation (e.g., content blocked). Error: {e}", exc_info=False) # No need for full stack trace for this
            raise # Re-raise to be handled by the router
        except Exception as e: # Catch-all for unexpected errors
            logger.critical(f"GeminiService: An unexpected error occurred during Gemini call. Error: {e}", exc_info=True)
            raise ConnectionError(f"An unexpected internal error occurred while contacting the Gemini service: {str(e)}") from e

    def _construct_prompt(self, idea_data: IdeaInput) -> str:
        """Helper method to construct the detailed prompt for Gemini."""
        # This prompt is highly influential on the output quality.
        return (
            f"You are an expert startup consultant and product manager. "
            f"Generate a comprehensive, actionable, and professionally formatted MVP (Minimum Viable Product) "
            f"scope document for the following startup idea:\n\n"
            f"**Industry:** {idea_data.industry}\n"
            f"**Core Idea:** {idea_data.idea}\n\n"
            f"**Instructions for the Scope Document:**\n"
            f"Please structure the document with clear, distinct sections. Use markdown-style formatting for headers (e.g., ## Section Title) and bullet points where appropriate.\n\n"
            f"**Include the following sections in detail:**\n\n"
            f"## 1. Executive Summary\n"
            f"   - A concise overview of the product, its core value proposition, and target audience.\n\n"
            f"## 2. Problem Statement & Proposed Solution\n"
            f"   - Clearly define the problem this startup aims to solve.\n"
            f"   - Describe how the proposed MVP addresses this problem.\n\n"
            f"## 3. Target Audience & User Personas\n"
            f"   - Identify the primary target users.\n"
            f"   - Briefly describe 1-2 key user personas (e.g., name, role, goals, pain points relevant to the product).\n\n"
            f"## 4. Core MVP Features (Prioritized)\n"
            f"   - List the essential features required for the MVP.\n"
            f"   - For each feature, provide a brief description.\n"
            f"   - Prioritize features using a simple scheme (e.g., P0 - Must-have, P1 - Should-have for V1, P2 - Nice-to-have/Future).\n\n"
            f"## 5. Key User Stories (for P0 Features)\n"
            f"   - Write 3-5 key user stories for the P0 (Must-have) features in the format: \"As a [user type/persona], I want to [perform an action] so that I can [achieve a benefit].\"\n\n"
            f"## 6. Technology Stack Recommendation (High-Level)\n"
            f"   - Suggest a suitable technology stack for the MVP (e.g., Frontend, Backend, Database, Key Cloud Services/APIs).\n"
            f"   - Briefly justify choices if specific (e.g., 'Python/FastAPI for rapid backend development').\n\n"
            f"## 7. Monetization Strategy (Initial Thoughts)\n"
            f"   - Outline 1-2 potential revenue models for the product (e.g., subscription, freemium, one-time purchase, transactional fees).\n\n"
            f"## 8. MVP Development Timeline (Estimated Phases)\n"
            f"   - Provide a high-level estimated timeline for MVP development, broken into logical phases or sprints (e.g., Phase 1 (4-6 weeks): Core Feature X, Y; User Authentication).\n\n"
            f"## 9. Key Metrics for Success (KPIs for MVP)\n"
            f"   - Define 3-5 measurable Key Performance Indicators (KPIs) to evaluate the MVP's success post-launch (e.g., user acquisition rate, daily active users, conversion rate for a key action, user retention rate).\n\n"
            f"## 10. Potential Risks & Mitigation Strategies\n"
            f"    - Identify 2-3 significant potential risks for the MVP (technical, market, operational).\n"
            f"    - Suggest a brief mitigation strategy for each identified risk.\n\n"
            f"Ensure the output is well-organized, easy to read, and provides practical, actionable insights."
        )

    def _get_generation_config(self) -> dict:
        """Helper method to return the generation configuration for Gemini."""
        return {
            "temperature": 0.6, # Lower for more factual, higher for more creative. 0.6 is a good balance.
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096, # Increased for potentially longer documents. Adjust based on model and needs.
            "candidate_count": 1, # We only need one candidate for this use case
        }

    def _get_safety_settings(self) -> dict:
        """Helper method to return safety settings for Gemini."""
        # These settings block content at a medium threshold. Adjust as needed.
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    def _process_gemini_response(self, response) -> str:
        """Helper method to process the response from Gemini API."""
        # Check for blocked content due to safety settings or other reasons
        if not response.candidates:
            block_reason_msg = "Content generation blocked by AI provider."
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = (f"Content generation blocked due to safety policy: "
                                    f"{response.prompt_feedback.block_reason.name}. "
                                    f"Consider revising the input idea or industry if it might violate content policies.")
            logger.warning(f"GeminiService: {block_reason_msg}")
            raise ValueError(block_reason_msg) # This will be caught by the router and returned as 400

        # Assuming one candidate as per generation_config
        try:
            generated_text = response.text.strip()
            if not generated_text: # Check for empty text even if candidate exists
                logger.warning("GeminiService: Received an empty text response from Gemini.")
                raise ValueError("AI service returned an empty response.")
            logger.info(f"GeminiService: Successfully processed Gemini response. Content length: {len(generated_text)}")
            return generated_text
        except AttributeError: # If response.text is not available for some reason (e.g. malformed response)
            logger.error("GeminiService: Malformed response from Gemini, 'text' attribute missing.", exc_info=True)
            # Log the full response for debugging if possible and not too large
            # logger.debug(f"Full Gemini response: {response}")
            raise ValueError("AI service returned a malformed response.")


# Instantiate the service globally for easy import into routers.
# This ensures the service is initialized once when the application starts.
gemini_service = GeminiService()