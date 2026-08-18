 # Criminal Face Detection System

This repository contains a Flask API (`main.py`) and a Streamlit frontend (`streamlit_app.py`) for a face-detection-based criminal registry. The project is prepared for local development and production deployment (Docker / Heroku).

**What I changed:** pinned dependencies, added `Dockerfile`, `Procfile`, `.dockerignore`, `init_db.py`, and improved deployment instructions.

## Quick start (local)

1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the database (optional — `main.py` will initialize on first run):

```bash
python init_db.py
```

4. Run the API locally:

```bash
python main.py
```

5. (Optional) Run the Streamlit UI locally in a separate terminal:

```bash
streamlit run streamlit_app.py
```

## Docker

Build and run with Docker:

```bash
docker build -t motu-app .
docker run -p 5000:5000 -e PORT=5000 --rm motu-app
```

## Heroku / Procfile

This repo contains a `Procfile` to run the app with `gunicorn` on Heroku. Basic steps:

```bash
heroku create my-app
git push heroku main
heroku ps:scale web=1
```

Set required environment variables on the platform (see `.env.example` below).

## Deploy to Streamlit Community Cloud

You can deploy the Streamlit UI (`streamlit_app.py`) directly to Streamlit Community Cloud (share.streamlit.io).

1. Push your repository to GitHub (ensure `requirements.txt` is up to date).

```bash
git add .
git commit -m "Prepare for Streamlit deployment"
git push origin main
```

2. Go to https://share.streamlit.io, sign in, and choose **New app** → **From a GitHub repo**.
3. Select your repository, branch `main`, and set the **File** to `streamlit_app.py`.
4. Add any required secrets or environment variables under **Advanced settings** (use `MATCH_THRESHOLD`, `FRONTEND_ORIGIN`, etc.).

Notes:
- The app imports `main.py` for shared logic; the Flask server is not started on import. The Streamlit app uses functions provided by `main.py` directly.
- If Streamlit fails during startup due to OpenCV missing contrib modules, ensure the deployment installs `opencv-contrib-python` (present in `requirements.txt`).


## Environment variables

Create a `.env` file (not committed) or set env vars in your platform. See `.env.example` for names and defaults.

- `PORT` — port used by the webserver (PaaS provides this)
- `FLASK_DEBUG` — `1` to enable debug (development only)
- `FRONTEND_ORIGIN` — allowed CORS origin (default `*` for development)
- `MATCH_THRESHOLD` — float 0..1 threshold used for face matching (default `0.6`)

## Security & storage notes

- `criminal_database.db` and `uploads/` are ignored by `.gitignore`. If the DB was previously committed, remove it from git history or untrack it:

```bash
git rm --cached criminal_database.db
git commit -m "Remove DB from repo"
```

- Use a managed database or mount a persistent volume for `uploads/` when deploying to production. The app stores files under the `uploads/` directory by default.

## Healthcheck endpoint

The simple root endpoint (`/`) returns API metadata and is suitable as a basic healthcheck. You can use `/` for container orchestration health probes.

## Initialize DB without running server

To create the database schema without starting the Flask server, run:

```bash
python init_db.py
```

## Troubleshooting

- If `opencv` fails to load Haar cascades, ensure `opencv-contrib-python` is installed in the same environment.
- If face detection reports no faces, test with a clear frontal image and increase image resolution.

## Next steps (optional)

- Add CI (GitHub Actions) to lint, run unit tests, and build the Docker image.
- Replace local SQLite with a managed DB for production scale.


