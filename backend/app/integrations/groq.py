"""Groq API integration for digest generation"""
from groq import Groq, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import os
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self, api_key: str = None):
        """Initialize Groq client with custom httpx client"""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")

        try:
            # Create custom httpx client to avoid version compatibility issues
            http_client = httpx.Client(timeout=30.0)
            self.client = Groq(api_key=self.api_key, http_client=http_client)
        except Exception as e:
            # Fallback: try without custom http_client
            logger.warning(f"Failed to create custom httpx client: {e}, using default")
            self.client = Groq(api_key=self.api_key)

        self.circuit_breaker_open = False
        self.circuit_breaker_attempts = 0
        self.max_failures = 3

    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError, Exception)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def create_completion(
        self,
        messages: list,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Call Groq API with retry logic and circuit breaker"""
        try:
            if self.circuit_breaker_open:
                if self.circuit_breaker_attempts < 3:
                    raise RuntimeError("Circuit breaker is open")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Reset circuit breaker on success
            self.circuit_breaker_open = False
            self.circuit_breaker_attempts = 0

            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:
            self.circuit_breaker_attempts += 1
            if self.circuit_breaker_attempts >= self.max_failures:
                self.circuit_breaker_open = True
                logger.error(f"Circuit breaker opened after {self.max_failures} failures: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return not self.circuit_breaker_open or self.circuit_breaker_attempts < self.max_failures

