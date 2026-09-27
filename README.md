# Course Catalog API

Small FastAPI service that defines a `Course` model and exposes a course-list endpoint at `GET /courses`.

## Run

Install the packages listed in `requirements.txt`, then start the application from the repository root:

```bash
uvicorn main:app --reload
```

Interactive API documentation is available at `/docs` while the service is running.
