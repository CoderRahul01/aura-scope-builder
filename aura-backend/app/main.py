# app/main.py
import logging
import logging.config
import sys
from contextlib import asynccontextmanager # For lifespan events if needed later

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings # This line implicitly loads settings
from app.routers import idea_router
from app.models.idea_models import ErrorResponse, ErrorDetail # For custom validation response

# --- Logging Configuration ---
# Moved to a separate function for clarity or can be a separate module
def setup_logging(log_level: str = "INFO"):
    """Configures structured logging for the application."""
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter", # Use Uvicorn's formatter for consistency
                "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(module)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": { # Renamed from 'console' for clarity
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": sys.stdout, # Explicitly use stdout
            },
            "access": { # Handler for Uvicorn access logs
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "root": {"handlers": ["default"], "level": log_level}, # Root logger
            "uvicorn": {"handlers": ["default"], "level": log_level, "propagate": False},
            "uvicorn.error": {"level": log_level, "handlers": ["default"], "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False}, # Access logs usually at INFO
            "app": {"handlers": ["default"], "level": "DEBUG" if settings.ENVIRONMENT != "production" else "INFO", "propagate": False},
            # Specific service loggers if needed more granular control
            "app.services.gemini_service": {"handlers": ["default"], "level": "DEBUG" if settings.ENVIRONMENT != "production" else "INFO", "propagate": True},
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)

# Determine log level based on environment
APP_LOG_LEVEL = "DEBUG" if settings.ENVIRONMENT == "development" else "INFO"
setup_logging(log_level=APP_LOG_LEVEL) # Call logging setup
logger = logging.getLogger("app.main") # Get a logger specific to this module

# --- FastAPI Lifespan Events (Optional) ---
# Example: For resource initialization/cleanup on startup/shutdown
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Application lifespan: Startup sequence initiated.")
#     # --- Initialize resources here (e.g., database connections, AI clients if not global) ---
#     # if not gemini_service.model: # Example re-check or specific init
#     #     logger.warning("Attempting to re-initialize Gemini service during lifespan startup.")
#     #     gemini_service_instance = GeminiService() # Re-instantiate if needed
#     #     if not gemini_service_instance.model:
#     #         logger.error("Failed to initialize Gemini service during lifespan.")
#     yield
#     # --- Cleanup resources here ---
#     logger.info("Application lifespan: Shutdown sequence initiated.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.2.0", # Corresponds to pyproject.toml
    description=(
        "API backend for generating Minimum Viable Product (MVP) scope documents "
        "for startup ideas using Google's Gemini Large Language Model."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs", # Standard interactive API documentation (Swagger UI)
    redoc_url="/redoc", # Alternative API documentation (ReDoc)
    # lifespan=lifespan, # Uncomment if using lifespan events
)

# --- Middleware Configuration ---
# CORS (Cross-Origin Resource Sharing)
# Adjust origins as per your frontend application's deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else ["https://your-frontend-domain.com"], # Be specific in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Specify allowed methods
    allow_headers=["*"], # Or specify allowed headers
)

# --- Custom Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for Pydantic's RequestValidationError to provide a more structured error response.
    """
    logger.warning(f"Request validation error for URL '{request.url}'. Errors: {exc.errors()}", exc_info=False)
    error_details = [
        ErrorDetail(loc=err.get("loc"), msg=err.get("msg"), type=err.get("type"))
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(detail=error_details).model_dump(),
    )

@app.exception_handler(Exception) # Generic catch-all for unhandled exceptions
async def general_exception_handler(request: Request, exc: Exception):
    """
    Custom handler for any unhandled exceptions to ensure a consistent JSON error response.
    """
    logger.error(f"Unhandled global exception for URL '{request.url}'. Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="An unexpected internal server error occurred. The technical team has been notified.").model_dump(),
    )

# --- API Routers ---
app.include_router(
    idea_router.router,
    prefix=settings.API_V1_STR,
    # tags=["AI Scope Generation"] # Tags can also be defined at the router level
)

# --- Root Endpoint & Health Check ---
@app.get("/", tags=["General"], summary="API Root Endpoint", response_model=dict)
async def read_root():
    """Provides basic information about the API."""
    logger.info(f"Root endpoint '/' accessed for '{settings.PROJECT_NAME}'.")
    return {
        "project_name": settings.PROJECT_NAME,
        "version": app.version,
        "environment": settings.ENVIRONMENT,
        "message": f"Welcome! API is operational. Visit '{app.docs_url}' or '{app.redoc_url}' for documentation.",
        "api_prefix": settings.API_V1_STR
    }

@app.get("/health", tags=["General"], summary="API Health Check", response_model=dict)
async def health_check():
    """Performs a basic health check of the API."""
    # In a real application, this could check database connections, external service status, etc.
    logger.debug("Health check endpoint '/health' accessed.")
    return {"status": "healthy", "message": "API is operational and ready to serve requests."}

# --- Application Startup Logging ---
logger.info(f"'{settings.PROJECT_NAME}' (v{app.version}) application startup complete.")
logger.info(f"Running in '{settings.ENVIRONMENT}' mode. Log level: {APP_LOG_LEVEL}.")
logger.info(f"Access OpenAPI docs at '{app.docs_url}' or ReDoc at '{app.redoc_url}'. API prefix: '{settings.API_V1_STR}'.")

# To run programmatically (less common for Uvicorn, usually run via CLI):
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8000,
#         log_level=APP_LOG_LEVEL.lower(), # Ensure uvicorn gets a lowercase log level string
#         reload=(settings.ENVIRONMENT == "development")
#     )