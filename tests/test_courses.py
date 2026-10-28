"""Pre-loaded sample courses/assignments: a static, read-only raw-dump
browser -- no LLM, no embeddings, no retrieval. See
core/course_service.py's module docstring for why the data is
fictional sample content, not a real AUM course feed."""

import pytest
from fastapi.testclient import TestClient

import core.course_service as course_service
import server.api_server as api_server


# ---------- core/course_service.py ----------

def test_list_courses_returns_summaries_without_assignment_content():
    courses = course_service.list_courses()
    assert len(courses) >= 1
    for course in courses:
        assert set(course.keys()) == {"course_id", "course_name", "instructor", "term", "assignment_count"}
        assert course["assignment_count"] >= 1


def test_list_assignments_returns_summaries_without_content():
    courses = course_service.list_courses()
    course_id = courses[0]["course_id"]
    assignments = course_service.list_assignments(course_id)
    assert len(assignments) == courses[0]["assignment_count"]
    for a in assignments:
        assert set(a.keys()) == {"assignment_id", "title", "due_date", "points"}
        assert "content" not in a  # the whole point: a listing never carries the dump


def test_list_assignments_raises_key_error_for_an_unknown_course():
    with pytest.raises(KeyError):
        course_service.list_assignments("NOT_A_REAL_COURSE")


def test_get_assignment_returns_the_full_raw_content():
    courses = course_service.list_courses()
    course_id = courses[0]["course_id"]
    assignment_id = course_service.list_assignments(course_id)[0]["assignment_id"]

    full = course_service.get_assignment(course_id, assignment_id)
    assert full["assignment_id"] == assignment_id
    assert full["course_id"] == course_id
    assert full["course_name"] == courses[0]["course_name"]
    assert isinstance(full["content"], str) and len(full["content"]) > 0


def test_get_assignment_raises_key_error_for_an_unknown_assignment():
    course_id = course_service.list_courses()[0]["course_id"]
    with pytest.raises(KeyError):
        course_service.get_assignment(course_id, "NOT_A_REAL_ASSIGNMENT")


def test_get_assignment_raises_key_error_for_an_unknown_course():
    with pytest.raises(KeyError):
        course_service.get_assignment("NOT_A_REAL_COURSE", "whatever")


# ---------- /api/courses endpoints ----------

def test_api_list_courses():
    client = TestClient(api_server.app)
    response = client.get("/api/courses")
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 1
    assert "course_id" in body[0] and "content" not in body[0]


def test_api_list_assignments():
    client = TestClient(api_server.app)
    course_id = client.get("/api/courses").json()[0]["course_id"]
    response = client.get(f"/api/courses/{course_id}/assignments")
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 1
    assert "content" not in body[0]


def test_api_list_assignments_404_for_unknown_course():
    client = TestClient(api_server.app)
    response = client.get("/api/courses/NOT_A_REAL_COURSE/assignments")
    assert response.status_code == 404


def test_api_get_assignment_dumps_full_content():
    client = TestClient(api_server.app)
    course_id = client.get("/api/courses").json()[0]["course_id"]
    assignment_id = client.get(f"/api/courses/{course_id}/assignments").json()[0]["assignment_id"]

    response = client.get(f"/api/courses/{course_id}/assignments/{assignment_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["assignment_id"] == assignment_id
    assert len(body["content"]) > 0


def test_api_get_assignment_404_for_unknown_assignment():
    client = TestClient(api_server.app)
    course_id = client.get("/api/courses").json()[0]["course_id"]
    response = client.get(f"/api/courses/{course_id}/assignments/NOT_REAL")
    assert response.status_code == 404
