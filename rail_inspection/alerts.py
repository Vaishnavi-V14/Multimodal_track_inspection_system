"""Telegram and email alert dispatch for inspection events."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


def send_telegram_message(bot_token: str, chat_id: str, text: str, timeout: float = 15.0) -> bool:
    if not bot_token or not chat_id:
        LOGGER.warning("Telegram skipped: missing bot_token or chat_id.")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4096], "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Telegram send failed: %s", exc)
        return False


def send_email_smtp(
    host: str,
    port: int,
    username: str,
    password: str,
    mail_from: str,
    mail_to: str,
    subject: str,
    body: str,
    use_tls: bool = True,
) -> bool:
    if not all([host, username, password, mail_from, mail_to]):
        LOGGER.warning("Email skipped: incomplete SMTP configuration.")
        return False
    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.sendmail(mail_from, mail_to.split(","), msg.as_string())
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Email send failed: %s", exc)
        return False


def load_alert_settings_from_streamlit_secrets(secrets: Any) -> dict[str, Any]:
    """Normalize st.secrets into flat keys if nested."""
    out: dict[str, Any] = {}
    if secrets is None:
        return out
    try:
        if hasattr(secrets, "get"):
            tg = secrets.get("telegram") or {}
            mail = secrets.get("smtp") or {}
            out["telegram_bot_token"] = tg.get("bot_token") or secrets.get("TELEGRAM_BOT_TOKEN")
            out["telegram_chat_id"] = tg.get("chat_id") or secrets.get("TELEGRAM_CHAT_ID")
            out["smtp_host"] = mail.get("host") or secrets.get("SMTP_HOST")
            out["smtp_port"] = int(mail.get("port") or secrets.get("SMTP_PORT") or 587)
            out["smtp_user"] = mail.get("user") or secrets.get("SMTP_USER")
            out["smtp_password"] = mail.get("password") or secrets.get("SMTP_PASSWORD")
            out["smtp_from"] = mail.get("from") or secrets.get("ALERT_EMAIL_FROM")
            out["smtp_to"] = mail.get("to") or secrets.get("ALERT_EMAIL_TO")
    except Exception:  # noqa: BLE001
        pass
    return out
