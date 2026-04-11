"""Email Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import EmailAction, EmailObservation, EmailState


class EmailEnvClient(EnvClient[EmailAction, EmailObservation, EmailState]):
    """
    Client for the Email Triage Environment.

    Maintains a persistent connection to the environment server, enabling
    multi-step inbox triage interactions.

    Example::

        from client import EmailEnvClient
        from models import EmailAction

        async with EmailEnvClient(base_url="http://localhost:7860") as env:
            result = await env.reset(task_name="medium", seed=42)
            while not result.done:
                obs = result.observation
                action = EmailAction(
                    category="billing",
                    urgency="medium",
                    action="respond",
                )
                result = await env.step(action)
                print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: EmailAction) -> Dict:
        """Convert EmailAction to JSON payload."""
        payload = {
            "category": action.category,
            "urgency": action.urgency,
            "action": action.action,
        }
        if action.reasoning:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)
        observation = EmailObservation(
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            sender_domain=obs_data.get("sender_domain", ""),
            received_at=obs_data.get("received_at", ""),
            thread_id=obs_data.get("thread_id", ""),
            thread_position=obs_data.get("thread_position", 1),
            thread_context=obs_data.get("thread_context", ""),
            emails_remaining=obs_data.get("emails_remaining", 0),
            emails_processed=obs_data.get("emails_processed", 0),
            queue_summary=obs_data.get("queue_summary", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EmailState:
        """Parse server response into EmailState."""
        return EmailState(
            task_name=payload.get("task_name", "easy"),
            emails_total=payload.get("emails_total", 0),
            emails_processed=payload.get("emails_processed", 0),
            current_thread_id=payload.get("current_thread_id", ""),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
