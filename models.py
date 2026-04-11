"""
Data models for the Email Triage Environment.

The email triage environment simulates a real-world IT support inbox where
an AI agent must classify, prioritize, and route incoming emails. Each
episode presents a queue of emails; the agent triages them one by one.
"""

from typing import List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server import Action, Observation, State


class EmailAction(Action):
    """Agent's triage decision for the current email.

    category:  spam | billing | technical | general | security
    urgency:   low | medium | high
    action:    mark_spam | archive | respond | escalate | escalate_urgent
    reasoning: optional free-text rationale (tracked, not graded)
    """

    category: str = Field(
        ..., description="Email category: spam, billing, technical, general, or security"
    )
    urgency: str = Field(
        ..., description="Urgency level: low, medium, or high"
    )
    action: str = Field(
        ..., description="Routing action: mark_spam, archive, respond, escalate, or escalate_urgent"
    )
    reasoning: str = Field(
        default="",
        description="Optional agent reasoning for the classification (not scored)",
    )


class EmailObservation(Observation):
    """Observation returned by the email triage environment.

    On reset(): presents the first email in the queue.
    On step():  returns the next email or done=True when the queue is empty.
    """

    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text")
    sender: str = Field(default="", description="Sender email address")
    sender_domain: str = Field(default="", description="Sender domain (e.g. company.com)")
    received_at: str = Field(default="", description="ISO‑8601 receive timestamp")
    thread_id: str = Field(default="", description="Conversation thread identifier")
    thread_position: int = Field(default=1, description="Position in thread (1 = first)")
    thread_context: str = Field(
        default="",
        description="Summary of prior emails in this thread (empty if first in thread)",
    )
    emails_remaining: int = Field(default=0, description="Emails left in the queue")
    emails_processed: int = Field(default=0, description="Emails already processed")
    queue_summary: str = Field(
        default="",
        description="Human-readable queue overview, e.g. '7 emails, 3 high priority'",
    )


class EmailState(State):
    """Server-side state for the email triage session."""

    task_name: str = Field(default="easy", description="Current task difficulty")
    emails_total: int = Field(default=0, description="Total emails in episode queue")
    emails_processed: int = Field(default=0, description="Emails processed so far")
    current_thread_id: str = Field(default="", description="Thread ID of current email")
    cumulative_reward: float = Field(default=0.0, description="Sum of rewards so far")


__all__ = ["EmailAction", "EmailObservation", "EmailState"]
