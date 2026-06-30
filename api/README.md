# Sanjaya API

FastAPI backend for the MMOU viewer.

Run it from the repository root:

```bash
just api
```

Default URL:

```text
http://localhost:8000
```

## Routes

- `GET /health`
- `GET /mmou-jobs/catalog`
- `POST /mmou-jobs`
- `GET /mmou-jobs`
- `GET /mmou-jobs/{job_id}`
- `POST /mmou-jobs/{job_id}/stop`
- `POST /mmou-jobs/{job_id}/resume`
- `POST /mmou-jobs/{job_id}/evaluate`
- `POST /mmou-jobs/{job_id}/questions/{question_id}/evaluate`
- `GET /mmou-jobs/{job_id}/questions/{question_id}/trace`
- `GET /mmou-jobs/{job_id}/events`
