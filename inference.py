"""
Inference Script — Email Triage Environment
============================================

MANDATORY VARIABLES (set via environment):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  (optional) Docker image name when using from_docker_image().

Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

from client import EmailEnvClient
from models import EmailAction

# ── Configuration ─────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["easy", "medium", "hard"]
BENCHMARK_NAME = "email_triage_env"
MAX_STEPS = 10  # safety cap per episode

VALID_CATEGORIES = {"spam", "billing", "technical", "general", "security"}
VALID_URGENCY = {"low", "medium", "high"}
VALID_ACTIONS = {"mark_spam", "archive", "respond", "escalate", "escalate_urgent"}

SYSTEM_PROMPT = """\
You are an expert email triage assistant for an IT support team. Your job is to classify incoming emails.

For each email, you must respond with ONLY a JSON object with these three fields:
{
  "category": "<spam|billing|technical|general|security>",
  "urgency": "<low|medium|high>",
  "action": "<mark_spam|archive|respond|escalate|escalate_urgent>"
}

Classification guidelines:
- category: What type of email is this?
  * spam: Unsolicited, phishing, scam, or junk email
  * billing: Invoices, payments, refunds, pricing, subscriptions
  * technical: Bugs, API issues, performance, feature requests, integrations
  * general: Internal comms, legal, partnerships, HR, policy
  * security: Data breaches, vulnerabilities, unauthorized access, compliance audits
- urgency: How time-sensitive is this?
  * low: Can wait days, no immediate impact
  * medium: Should be addressed within 24 hours
  * high: Needs immediate attention, significant business impact
- action: What should be done?
  * mark_spam: Flag as spam / junk
  * archive: File for records, no action needed
  * respond: Needs a standard response from the support team
  * escalate: Route to specialized team for handling
  * escalate_urgent: Route immediately to on-call / management

Consider the sender domain, subject tone, body content, and any thread context.
Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""


# ── Helpers ───────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _normalize_action(payload: dict) -> EmailAction:
    """Normalize LLM output into a valid EmailAction."""
    category = str(payload.get("category", "general")).strip().lower()
    urgency = str(payload.get("urgency", "medium")).strip().lower()
    action = str(payload.get("action", "respond")).strip().lower()

    # Normalize common variations
    category = category.replace("-", "_").replace(" ", "_")
    urgency = urgency.replace("-", "_").replace(" ", "_")
    action = action.replace("-", "_").replace(" ", "_")

    # Map legacy/variant names
    if action == "ignore":
        action = "archive"
    if action == "flag_spam":
        action = "mark_spam"
    if action in ("escalate_immediately", "urgent_escalate"):
        action = "escalate_urgent"

    if category not in VALID_CATEGORIES:
        category = "general"
    if urgency not in VALID_URGENCY:
        urgency = "medium"
    if action not in VALID_ACTIONS:
        action = "respond"

    return EmailAction(category=category, urgency=urgency, action=action)


def _action_str(action: EmailAction) -> str:
    return f"cat:{action.category}|urg:{action.urgency}|act:{action.action}"


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


# ── LLM interaction ──────────────────────────────────────────────────────


def choose_action(
    client: OpenAI,
    model_name: str,
    subject: str,
    body: str,
    sender: str,
    sender_domain: str,
    thread_context: str,
    queue_summary: str,
) -> EmailAction:
    """Ask the LLM to classify an email and return a normalized action."""
    user_parts = [
        f"Subject: {subject}",
        f"From: {sender} ({sender_domain})",
        f"Body: {body}",
    ]
    if thread_context:
        user_parts.append(f"Thread context: {thread_context}")
    if queue_summary:
        user_parts.append(f"Queue status: {queue_summary}")

    user_msg = "\n".join(user_parts)

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = _extract_json(content)
        return _normalize_action(payload)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return EmailAction(category="general", urgency="medium", action="respond")


# ── Logging ───────────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    done_val = _bool_str(done)
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Episode runner ────────────────────────────────────────────────────────


async def _close_env(env_client: EmailEnvClient) -> None:
    try:
        await env_client.disconnect()
    except Exception:
        pass
    try:
        await env_client.close()
    except Exception:
        pass


async def run_episode(
    task: str,
    llm: OpenAI,
    model_name: str,
    env_base_url: str,
    local_image_name: Optional[str],
) -> None:
    """Run a single episode for a given task."""
    rewards: List[float] = []
    steps = 0
    success = False
    env_client: Optional[EmailEnvClient] = None

    log_start(task=task, env=BENCHMARK_NAME, model=model_name)

    try:
        if local_image_name:
            env_client = await EmailEnvClient.from_docker_image(local_image_name)
        else:
            env_client = EmailEnvClient(base_url=env_base_url)

        await env_client.connect()
        reset_result = await env_client.reset(task_name=task, seed=42)
        obs = reset_result.observation

        done = obs.done
        while not done and steps < MAX_STEPS:
            steps += 1

            action = choose_action(
                client=llm,
                model_name=model_name,
                subject=obs.subject,
                body=obs.body,
                sender=obs.sender,
                sender_domain=obs.sender_domain,
                thread_context=obs.thread_context,
                queue_summary=obs.queue_summary,
            )

            step_result = await env_client.step(action)
            reward_value = float(step_result.reward or 0.0)
            done = bool(step_result.done)
            rewards.append(reward_value)

            step_obs = getattr(step_result, "observation", None)
            raw_error = getattr(step_obs, "last_action_error", None)
            error_str = str(raw_error) if raw_error else None

            log_step(
                step=steps,
                action=_action_str(action),
                reward=reward_value,
                done=done,
                error=error_str,
            )

            if not done:
                obs = step_result.observation

        # Episode score = mean of per-step rewards, clamped [0, 1]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.3  # reasonable threshold for partial success

    except Exception as exc:
        steps = max(steps, 1)
        rewards = rewards or [0.0]
        log_step(
            step=steps, action="error", reward=0.0, done=False, error=str(exc)
        )
        success = False
        score = 0.0

    finally:
        if env_client is not None:
            await _close_env(env_client)
        s = sum(rewards) / len(rewards) if rewards else 0.0
        s = max(0.0, min(1.0, s))
        log_end(success=success, steps=steps, score=s, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────


async def run() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

    for task in TASKS:
        await run_episode(
            task=task,
            llm=llm,
            model_name=MODEL_NAME,
            env_base_url=env_base_url,
            local_image_name=LOCAL_IMAGE_NAME,
        )


if __name__ == "__main__":
    asyncio.run(run())
