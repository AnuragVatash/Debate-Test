# Debate Engine Webapp (Flask)

Quick Flask app to browse scored debates and render the Elo timeline.

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask flask-socketio
```

Tailwind (optional for styling):

```bash
# In webapp/
npm install -D tailwindcss autoprefixer postcss
npx tailwindcss -i ./static/src.css -o ./static/tailwind.css --watch
```

## Run

```bash
cd webapp
python app.py
# Visit http://localhost:5000
```

The app scans `../diatrize/*_scored.json` and lists them on the homepage.
Click a debate to view the timeline and speaker cards.

## Notes

- WebSocket channel is initialized; you can push deltas later from your scoring worker.
- Plotly renders the cumulative advantage line from the `timeline` array.
- Add auth, uploads, and background workers as needed.
