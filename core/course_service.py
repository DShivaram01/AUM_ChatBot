"""Pre-loaded sample courses and their assignments.

Explicitly fictional sample data (data/courses_data.json), not real AUM
course records -- same convention exp/quiz-map-reduce's own fictional
benchmark PDFs already used for this kind of demo content. Read-only,
static, no retrieval/embeddings/LLM: "dump the contents" is a literal
dump of whatever's in the JSON record, not a generated or summarized
view.
"""

from __future__ import annotations

import json

import config

_COURSES: list[dict] | None = None


def _load() -> list[dict]:
    global _COURSES
    if _COURSES is None:
        with open(config.COURSES_JSON, encoding="utf-8") as fh:
            _COURSES = json.load(fh)
    return _COURSES


def _get_course(course_id: str) -> dict:
    for course in _load():
        if course["course_id"] == course_id:
            return course
    raise KeyError(f"No course '{course_id}'")


def list_courses() -> list[dict]:
    """Course summaries only -- never assignment content from a listing
    endpoint (same discipline DocumentService.retrieve() already
    applies to its own chunk text)."""
    return [
        {
            "course_id": course["course_id"],
            "course_name": course["course_name"],
            "instructor": course.get("instructor", ""),
            "term": course.get("term", ""),
            "assignment_count": len(course.get("assignments", [])),
        }
        for course in _load()
    ]


def list_assignments(course_id: str) -> list[dict]:
    """Assignment summaries for one course -- title/due_date/points
    only, not content. Raises KeyError if course_id doesn't exist."""
    course = _get_course(course_id)
    return [
        {
            "assignment_id": a["assignment_id"],
            "title": a["title"],
            "due_date": a.get("due_date", ""),
            "points": a.get("points"),
        }
        for a in course.get("assignments", [])
    ]


def get_assignment(course_id: str, assignment_id: str) -> dict:
    """The full raw dump: everything stored for this assignment,
    content included, exactly as written -- no summarization, no
    chunking, no validation. Raises KeyError if the course or the
    assignment within it doesn't exist.
    """
    course = _get_course(course_id)
    for assignment in course.get("assignments", []):
        if assignment["assignment_id"] == assignment_id:
            return {
                **assignment,
                "course_id": course["course_id"],
                "course_name": course["course_name"],
            }
    raise KeyError(f"No assignment '{assignment_id}' in course '{course_id}'")
