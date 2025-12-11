"""
Main entry point for the LoL AI Analysis Service

This is the primary entry point for running the FastAPI application.
Similar to Spring Boot's main class with @SpringBootApplication.

Usage:
    Development mode (with auto-reload):
        python main.py

    Production mode:
        python main.py --env production

    Custom port:
        python main.py --port 8080
"""

import uvicorn
import argparse
import sys
from src.api.routes import app
from src.config.settings import settings
from src.config.log_config import LOGGING_CONFIG


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoL AI Analysis Service - FastAPI Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.API_HOST,
        help=f"Host to bind (default: {settings.API_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.API_PORT,
        help=f"Port to bind (default: {settings.API_PORT})",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="development",
        choices=["development", "production"],
        help="Environment mode (default: development)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Log level (default: {settings.LOG_LEVEL})",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (production mode)",
    )

    return parser.parse_args()


def main():
    """Main function to start the FastAPI server"""
    args = parse_args()

    # Determine reload based on environment
    reload = not args.no_reload and args.env == "development"

    print("=" * 60)
    print("LoL AI Analysis Service")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {reload}")
    print(f"Log level: {args.log_level}")
    print("-" * 60)
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60)

    try:
        uvicorn.run(
            "src.api.routes:app",
            host=args.host,
            port=args.port,
            reload=reload,
            log_level=args.log_level.lower(),
            access_log=True,
            log_config=LOGGING_CONFIG,
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFailed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
