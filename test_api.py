from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


# test 1 — health check
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


# test 2 — detect endpoint exists
def test_detect_missing_url():
    response = client.post("/detect", json={})
    assert response.status_code == 422  # validation error = correct


# test 3 — notes endpoint returns expected shape
def test_get_notes():
    response = client.get("/notes")
    assert response.status_code == 200
    data = response.json()
    assert "topics"  in data
    assert "journey" in data
    assert "stats"   in data