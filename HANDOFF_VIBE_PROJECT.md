# VieNeu-Inspired Personal TTS - Vibe Coding Handoff (Portable)

> This document is designed to be copied into a **completely empty folder** and used as the single source of truth to bootstrap the project from zero.

## Mission
- Build a **personal TTS project inspired by VieNeu-TTS** with the same architecture style.
- Prioritize **shipping fast, iterating fast, and keeping momentum**.
- Goal is not perfect code first; goal is a usable product that improves every session.

## Empty Folder Bootstrap (Start Here)
- Put this file at project root.
- Create initial folders:
1. `src/yourtts`
2. `src/yourtts/engines`
3. `src/yourtts/utils`
4. `apps`
5. `tests`
6. `assets`
- Create initial files:
1. `pyproject.toml`
2. `README.md`
3. `.gitignore`
4. `config.yaml`
5. `src/yourtts/__init__.py`
6. `src/yourtts/base.py`
7. `src/yourtts/factory.py`
8. `src/yourtts/engines/standard.py`
9. `apps/web_ui.py`
10. `apps/api.py`
11. `tests/test_factory.py`

## Day-0 Technical Goal
- By end of first setup session, you must be able to run:
1. `python -m yourtts.smoke` (or equivalent quick script)
2. Generate one short `.wav` file
3. Start web UI locally (even if output is dummy/test audio)

## Product Goal (What "Done" Means)
- A local app can:
1. Load one model backend.
2. Synthesize text to `.wav`.
3. Support at least one preset voice.
4. Expose a simple web UI.
5. Expose a minimal API endpoint for synthesis.

## Vibe Coding Rules (Non-Negotiable)
- 100% vibe coding:
1. Small steps, immediate feedback, no over-planning.
2. Run/test after each meaningful change.
3. Prefer working prototype over abstraction-heavy refactor.
4. Keep commits small and descriptive.
5. If stuck > 20 minutes, simplify scope and ship the smaller version.

## Architecture Target (Mirror VieNeu Logic)
- `factory` chooses engine mode.
- `base engine` contains shared logic (voice, prompt, save, common validation).
- per-engine implementations:
1. `standard` (first MVP engine)
2. `turbo` (phase 2)
3. `fast` / `remote` (later)
- separate app layers:
1. `apps/web_ui.py`
2. `apps/api.py`
- utility layer:
1. text chunking
2. phonemization
3. audio join/crossfade

## Minimum Dependencies (MVP)
- Keep dependency list minimal first; add only when needed:
1. `numpy`
2. `soundfile`
3. `pyyaml`
4. `gradio`
5. (optional for phase 1) model backend dependency you choose (`transformers` or `llama-cpp-python`)

## Scope Phases

### Phase 1 - MVP Core
- [ ] Create package skeleton (`src/yourtts/...`).
- [ ] Implement `BaseEngine` interface.
- [ ] Implement `Factory(mode=...)`.
- [ ] Implement one real engine (`standard`) with `infer()`.
- [ ] Save audio output via `soundfile`.
- [ ] Add a simple smoke script to validate full path quickly.

### Phase 2 - Usable Product
- [ ] Add preset voices (`voices.json`).
- [ ] Add text split + chunk join utilities.
- [ ] Add basic Gradio web UI.
- [ ] Add simple API (`POST /synthesize`).
- [ ] Add config file for model/device settings.

### Phase 3 - Quality & Scale
- [ ] Add batch inference.
- [ ] Add streaming inference (optional if time allows).
- [ ] Add caching + warmup.
- [ ] Add tests for factory, utils, and one engine path.
- [ ] Add Docker run path.

## Session Startup Checklist (For Any New Codex Agent)
- [ ] Read this file first.
- [ ] Check current repo status and recent edits.
- [ ] Confirm current active phase and next unchecked task.
- [ ] Continue from next smallest shippable task.
- [ ] Run smoke test before ending session.
- [ ] Update this file progress checkboxes + short session note.

## Golden Rule For New Machine / New Agent
- Do not assume any previous repo exists.
- Do not assume virtualenv exists.
- Do not assume model weights are downloaded.
- Start from this file and bootstrap the smallest runnable path first.

## Session End Template
- Date:
- What was shipped:
- What is working now:
- Known blockers:
- Next first task:

## Current Status (Initialize)
- Active phase: **Phase 1 - MVP Core**
- Next task: **Create package skeleton + BaseEngine + Factory in empty folder**
- Blockers: **None recorded yet**

## Definition of Momentum
- Each session must end with at least one of:
1. New feature working.
2. Existing flow more stable.
3. Clear blocker documented with next action.

If none of the above happened, scope is too big. Cut scope and ship smaller.
