import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "News Agent API" in response.json()["message"]

def test_api_docs():
    """Test API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "openapi" in response.headers.get("content-type", "").lower() or "html" in response.headers.get("content-type", "").lower()
