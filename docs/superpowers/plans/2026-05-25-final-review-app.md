# Final Review App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Build a Flask web app for reviewing all suspected herring spawn positives and creating a golden training set.

**Architecture:** Single Flask app (`scripts/final_review_app.py`) with embedded HTML template serving all 305 suspects from `data/final_review/suspects.json`. Three sections: Gold Candidates (24 accepted, expanded), Previously Rejected (281, collapsed), Missing Files (35, reference). Images served via Flask static route. Decisions tracked in memory + JSON file. Finalize button exports golden set and updates training manifest.

**Tech Stack:** Python 3, Flask, Jinja2, JSON

---

### Task 1: Create `scripts/final_review_app.py`

**Files:**
- Create: `scripts/final_review_app.py`

- [ ] **Step 1: Write the Flask app with all routes**

The app has these routes:
- `GET /` — main review page (renders Jinja2 template)
- `GET /api/suspects` — JSON endpoint returning filtered/sorted suspect list
- `POST /api/label` — record accept/reject/skip for a suspect
- `GET /api/stats` — return review progress stats
- `POST /api/finalize` — save golden set JSON and update training manifest
- `GET /thumbnail/<path:disk_path>` — serve image file from disk
- `POST /api/reset` — reset all decisions

Key design decisions:
- Load `suspects.json` at startup into a list of dicts
- Track decisions in a dict `{filename: "accept"|"reject"|"skip"}` saved to `data/final_review/decisions.json`
- Serve thumbnails via a route that reads from `disk_path` to handle scattered file locations
- The template is embedded as a multi-line string in the Python file to avoid external file dependencies

- [ ] **Step 2: Build the Jinja2 template**

The template includes:
- Dark theme CSS (embedded `<style>` block)
- Three sections: Gold Candidates, Previously Rejected, Missing Files
- Each suspect card shows: thumbnail, filename, status badge, reviewer labels (who said what), sources
- Sort controls: default, reviewer count, filename
- Filter controls: all, unlabeled, accepted, rejected
- Keyboard shortcuts: ← reject, → accept, ↓ skip
- Progress counter
- Finalize button
- All JavaScript embedded in `<script>` block

- [ ] **Step 3: Test the app starts and loads**

Run: `python scripts/final_review_app.py`
Verify: server starts on port 8781, loads suspects.json, shows correct counts

- [ ] **Step 4: Commit**

```bash
git add scripts/final_review_app.py
git commit -m "feat: add final review app for golden set creation"
```
