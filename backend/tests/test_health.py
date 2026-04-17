import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "News Agent API" in response.json()["message"]

def test_api_docs(client):
    """Test API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "openapi" in response.headers.get("content-type", "").lower() or "html" in response.headers.get("content-type", "").lower()
