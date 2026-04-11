---
title: Email Triage Environment Server
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /docs
tags:
  - openenv
---

# Email Triage Environment for OpenEnv

A multi-step, real-world email triage environment where an AI agent processes an inbox queue of support emails — classifying category, urgency, and routing action for each message. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Why Email Triage?

Email triage is one of the most common tasks in IT support and customer service. Support teams spend hours every day:

- Filtering spam from legitimate requests
- Identifying the correct department (billing, technical, security)
- Assessing urgency to prioritize critical issues
- Routing messages to the right team or escalation path

This environment captures that real-world complexity, including **thread chains** (follow-up emails referencing prior context), **deceptive phishing** that mimics legitimate alerts, and **time-sensitive incidents** requiring immediate escalation.

## Quick Start

### Docker (Recommended)

```bash
docker build -t email-triage-env:latest .
docker run --rm -p 7860:7860 email-triage-env:latest
curl http://localhost:7860/health
```

Health response: `{"status":"healthy"}`

### Without Docker

```bash
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Environment Design

### Episode Structure

Each episode is **multi-step** — the agent processes a queue of emails one at a time:

1. `reset(task_name="medium", seed=42)` → returns the first email
2. `step(action)` → scores the classification, returns the next email
3. Repeat until the queue is empty (`done=True`)

| Task | Queue Size | Description |
|------|-----------|-------------|
| `easy` | 5 emails | Obvious spam, clear-cut billing/technical |
| `medium` | 6 emails | Ambiguous classifications, thread follow-ups, disputes |
| `hard` | 7 emails | Critical incidents, sophisticated phishing, legal/compliance |

### Action Space

`EmailAction` — the agent's triage decision for each email:

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `category` | `str` | `spam`, `billing`, `technical`, `general`, `security` | Email category |
| `urgency` | `str` | `low`, `medium`, `high` | Time sensitivity |
| `action` | `str` | `mark_spam`, `archive`, `respond`, `escalate`, `escalate_urgent` | Routing decision |
| `reasoning` | `str` | free text | Optional agent rationale (not scored) |

### Observation Space

`EmailObservation` — what the agent sees for each email:

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `str` | Email subject line |
| `body` | `str` | Email body text |
| `sender` | `str` | Sender address |
| `sender_domain` | `str` | Sender domain (e.g. `company.com`) |
| `received_at` | `str` | ISO‑8601 timestamp |
| `thread_id` | `str` | Conversation thread ID |
| `thread_position` | `int` | Position in thread (1 = first message) |
| `thread_context` | `str` | Summary of prior emails in thread |
| `emails_remaining` | `int` | Emails left in queue |
| `emails_processed` | `int` | Emails already classified |
| `queue_summary` | `str` | Overview: "5 remaining, 2 high priority" |

### State

`EmailState` — server-side session state:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Current difficulty level |
| `emails_total` | `int` | Total emails in episode |
| `emails_processed` | `int` | Emails classified so far |
| `current_thread_id` | `str` | Active thread ID |
| `cumulative_reward` | `float` | Running reward total |

## Reward Function

Each step produces a shaped reward in `[0.0, 1.0]` based on classification accuracy:

### Base Rewards

| Component | Exact Match | Near Miss | Wrong |
|-----------|------------|-----------|-------|
| Category | +0.40 | +0.15 | 0.00 |
| Urgency | +0.25 | +0.10 (off-by-one) | 0.00 |
| Action | +0.25 | +0.10 | 0.00 |

**Maximum base reward per step: 0.90**

### Bonuses

| Bonus | Value | Condition |
|-------|-------|-----------|
| Thread consistency | +0.10 | Correct category matches prior classification in same thread |
| Thread partial | +0.05 | Category matches prior thread decision (even if wrong) |

### Penalties

| Penalty | Value | Condition |
|---------|-------|-----------|
| False spam | −0.20 | Marking a legitimate email as spam |
| Ignoring critical | −0.15 | Archiving/spam-marking a high-urgency email |

### Near-Miss Pairs

- **Category**: billing ↔ general, technical ↔ security, spam ↔ general
- **Action**: archive ↔ respond, escalate ↔ escalate_urgent, mark_spam ↔ archive

### Episode Score

The episode score is the **mean of per-step rewards**, clamped to `[0, 1]`.

## Email Corpus

The environment includes **50+ realistic emails** across all difficulty levels:

- **Easy**: Obvious spam (lottery, phishing, pills), clear receipts, routine notifications
- **Medium**: Billing disputes, API issues, feature requests, ToS updates, thread follow-ups
- **Hard**: Production outages, security breaches, GDPR/DMCA requests, executive escalations, sophisticated phishing with lookalike domains

Thread chains link related emails across the queue to test contextual reasoning.

## Baseline Inference

The required inference script is at project root: `inference.py`.

### Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # default
export HF_TOKEN="your-huggingface-token"                  # required
export ENV_BASE_URL="http://127.0.0.1:7860"               # optional
export LOCAL_IMAGE_NAME=""                                 # optional
```

### Run

```bash
uv run python inference.py
```

### Output Format

The script emits structured logs:

```
[START] task=easy env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=cat:spam|urg:low|act:mark_spam reward=0.90 done=false error=null
[STEP] step=2 action=cat:billing|urg:low|act:archive reward=0.90 done=false error=null
...
[END] success=true steps=5 score=0.88 rewards=0.90,0.90,0.85,0.90,0.85
```

## Project Structure

```
email-triage-env/
├── __init__.py         # Package exports
├── models.py           # EmailAction, EmailObservation, EmailState
├── client.py           # EmailEnvClient (EnvClient subclass)
├── inference.py        # Baseline inference script (mandatory)
├── openenv.yaml        # OpenEnv manifest
├── pyproject.toml      # Dependencies
├── uv.lock             # Locked dependencies
├── Dockerfile          # Container image
├── README.md           # This file
└── server/
    ├── __init__.py     # Server package
    ├── app.py          # FastAPI application
    └── environment.py  # Core environment logic + email corpus
```

## Deploying to Hugging Face Spaces

Push to a Docker-SDK HF Space:

```bash
openenv push --repo-id your-username/email-triage-env
```

Or manually: create a Docker Space, push the repo, and ensure `app_port: 7860` is set.

## Development

### Run environment tests locally

```bash
python server/environment.py
```

This runs a perfect-agent simulation across all three tasks and prints per-step rewards.

### Run the server for development

```bash
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
```

## References

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv/tree/main/docs)

## License

BSD-3-Clause
