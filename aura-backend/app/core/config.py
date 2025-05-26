# app/core/config.py
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional # For optional OpenAI settings

logger = logging.getLogger(__name__) # Standard practice to get logger by module name

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables and .env file.
    """
    PROJECT_NAME: str = "Aura MVP Scope Generator (Gemini Edition)"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development" # e.g., development, staging, production

    # --- Google Gemini Configuration ---
    GEMINI_API_KEY: str # This is now a required field. Startup will fail if not set.
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest" # Default model

    # --- OpenAI Configuration (Optional, if you plan to support both or switch back) ---
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini" # Updated default if kept
    OPENAI_MAX_TOKENS: int = 1500

    # Pydantic-settings configuration to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra fields found in environment or .env
        case_sensitive=False # Environment variables are often case-insensitive
    )

# Instantiate settings globally for easy access throughout the application.
# This block also performs an initial check for the critical GEMINI_API_KEY.
try:
    settings = Settings()
    logger.info(f"Application settings loaded successfully for '{settings.PROJECT_NAME}'. Environment: {settings.ENVIRONMENT}")

    if not settings.GEMINI_API_KEY:
        # This condition should ideally not be met if GEMINI_API_KEY is non-optional
        # Pydantic would raise a validation error before this during Settings() instantiation.
        # However, keeping a log for explicit clarity.
        logger.critical("CRITICAL: GEMINI_API_KEY is not set in the environment or .env file. Gemini AI features will NOT work.")
        # Consider raising an exception here to halt startup if this is absolutely critical.
        # raise ValueError("GEMINI_API_KEY is not configured.")
    else:
        logger.info("GEMINI_API_KEY found and loaded.")

    if settings.OPENAI_API_KEY:
        logger.info("OpenAI API Key found (optional feature).")
    else:
        logger.info("OpenAI API Key not found (optional feature not configured).")

except Exception as e: # Catch potential Pydantic validation errors during instantiation
    logger.critical(f"CRITICAL ERROR: Could not load application settings. Error: {e}", exc_info=True)
    # Re-raising is important to halt application startup if config is invalid.
    raise ValueError(f"Fatal error initializing application settings: {e}") from e