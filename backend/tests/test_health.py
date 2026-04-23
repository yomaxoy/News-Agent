import pytest

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["environment"] == "development"
    assert "is_production" in data
    assert data["is_production"] == False
    assert "cookie_samesite" in data
    assert "cookie_secure" in data
    assert "cors_allow_origins" in data

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
