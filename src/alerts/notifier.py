"""
Alert Notifier
==============

Delivers surge alerts through pluggable channels. The default
:class:`LoggingChannel` requires no credentials and simply logs, so the whole
system runs out of the box. Optional channels (Slack webhook, SMTP email) are
activated only when their configuration is supplied, and they fail soft.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import List, Optional, Protocol, Sequence

from src.alerts.early_warning import Alert, EarlyWarningSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_alerts(alerts: Sequence[Alert]) -> str:
    """Render alerts as a short human-readable message."""
    actionable = EarlyWarningSystem.actionable(alerts)
    if not actionable:
        return "No ER demand surge predicted."
    worst = EarlyWarningSystem.highest_severity(actionable)
    lines = [f"ER demand surge alert (highest severity: {worst.upper()})", ""]
    for a in actionable:
        lines.append(
            f"  {a.date}: {a.predicted_visits:.0f} visits "
            f"(+{a.expected_excess:.0f} vs baseline {a.baseline_mean:.0f}, "
            f"z={a.z_score:.1f}) -> {a.severity.upper()}"
        )
    return "\n".join(lines)


class NotificationChannel(Protocol):
    """A sink that delivers a formatted alert message."""

    def send(self, subject: str, message: str) -> bool: ...


class LoggingChannel:
    """Default channel: log the alert. Always available, no configuration."""

    def __init__(self):
        self.logger = logger

    def send(self, subject: str, message: str) -> bool:
        self.logger.info("[ALERT] %s\n%s", subject, message)
        return True


class SlackWebhookChannel:
    """Post alerts to a Slack incoming webhook (if a URL is configured)."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def send(self, subject: str, message: str) -> bool:
        if not self.webhook_url:
            logger.warning("SlackWebhookChannel: no webhook URL configured; skipping")
            return False
        try:
            import requests

            resp = requests.post(
                self.webhook_url,
                json={"text": f"*{subject}*\n```{message}```"},
                timeout=10,
            )
            return resp.status_code < 300
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("SlackWebhookChannel failed: %s", exc)
            return False


class EmailChannel:
    """Send alerts via SMTP (if SMTP settings are configured in the env)."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[Sequence[str]] = None,
    ):
        self.host = host or os.environ.get("SMTP_HOST")
        self.port = int(os.environ.get("SMTP_PORT", port))
        self.username = username or os.environ.get("SMTP_USERNAME")
        self.password = password or os.environ.get("SMTP_PASSWORD")
        self.sender = sender or os.environ.get("ALERT_EMAIL_FROM")
        env_recipients = os.environ.get("ALERT_EMAIL_TO", "")
        self.recipients = list(
            recipients or [r for r in env_recipients.split(",") if r]
        )

    def send(self, subject: str, message: str) -> bool:
        if not (self.host and self.sender and self.recipients):
            logger.warning("EmailChannel: SMTP not fully configured; skipping")
            return False
        try:  # pragma: no cover - network dependent
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("EmailChannel failed: %s", exc)
            return False


class AlertNotifier:
    """Fan an alert message out to one or more channels."""

    def __init__(self, channels: Optional[List[NotificationChannel]] = None):
        self.channels: List[NotificationChannel] = channels or [LoggingChannel()]

    def notify(
        self, alerts: Sequence[Alert], subject: str = "ER Demand Early Warning"
    ) -> List[bool]:
        """Deliver alerts to every channel; returns per-channel success flags."""
        message = format_alerts(alerts)
        return [channel.send(subject, message) for channel in self.channels]
