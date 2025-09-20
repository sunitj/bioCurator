"""Simple health check implementation."""

import sys
import asyncio
import httpx

async def check_health(base_url: str = "http://localhost:8080") -> bool:
    """Simple health check for the application."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health/")
            return response.status_code == 200
    except Exception:
        return False

def main():
    """CLI health check."""
    result = asyncio.run(check_health())
    if result:
        print("Health check: PASSED")
        sys.exit(0)
    else:
        print("Health check: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()