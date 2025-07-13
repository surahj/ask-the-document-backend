import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class EmailServiceProvider:
    """Base email service provider interface."""

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        raise NotImplementedError


class GmailService(EmailServiceProvider):
    """Gmail SMTP email service implementation matching Go backend."""

    def __init__(self, sender_email: str, app_password: str):
        self.sender_email = sender_email
        self.app_password = app_password
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email using Gmail SMTP server."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = to
            msg["Subject"] = subject

            # Attach HTML body
            msg.attach(MIMEText(body, "html"))

            # Connect to Gmail SMTP server
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()  # Enable TLS encryption
            server.login(self.sender_email, self.app_password)

            # Send email
            text = msg.as_string()
            server.sendmail(self.sender_email, to, text)
            server.quit()

            logger.info(f"Email sent successfully to {to}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to}: {str(e)}")
            return False


class LogEmailService(EmailServiceProvider):
    """Log email service for development - backend."""

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Log email details instead of sending."""
        print("\n--- New Email ---")
        print(f"To: {to}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print("--- End Email ---\n")
        return True


class EmailService:
    """Email service factory matching backend."""

    def __init__(self):
        self._provider: Optional[EmailServiceProvider] = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize email provider based on environment configuration."""
        provider = settings.email_provider or os.getenv("EMAIL_PROVIDER", "log")

        if provider == "gmail":
            sender_email = settings.gmail_sender_email or os.getenv(
                "GMAIL_SENDER_EMAIL"
            )
            app_password = settings.gmail_app_password or os.getenv(
                "GMAIL_APP_PASSWORD"
            )

            if not sender_email or not app_password:
                logger.warning(
                    "GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD must be set for gmail provider. Using log provider."
                )
                self._provider = LogEmailService()
            else:
                self._provider = GmailService(sender_email, app_password)
                logger.info("Gmail email service initialized")
        else:
            # Default to log provider for development
            logger.info(
                f"Email provider '{provider}' not configured or unknown, using log provider."
            )
            self._provider = LogEmailService()

    async def send_otp_email(self, email: str, first_name: str, otp: str) -> bool:
        """Send OTP verification email."""
        subject = "Your Documind OTP"
        body = f"""
        <html>
            <body>
                <p>Hi {first_name},</p>
                <br>
                <p>Your One-Time Password (OTP) for Documind is: <strong>{otp}</strong></p>
                <br>
                <p>This OTP is valid for 10 minutes. Please use it to complete your registration.</p>
                <br>
                <p>Thanks,<br>The Documind Team</p>
            </body>
        </html>
        """
        return await self._provider.send_email(email, subject, body)

    async def send_password_reset_email(
        self, email: str, first_name: str, otp: str
    ) -> bool:
        """Send password reset email."""
        subject = "Your Password Reset OTP"
        body = f"""
        <html>
            <body>
                <p>Hi {first_name},</p>
                <br>
                <p>Your password reset OTP for Documind is: <strong>{otp}</strong></p>
                <br>
                <p>This OTP is valid for 10 minutes. Please use it to reset your password.</p>
                <br>
                <p>If you didn't request this password reset, please ignore this email.</p>
                <br>
                <p>Thanks,<br>The Documind Team</p>
            </body>
        </html>
        """
        return await self._provider.send_email(email, subject, body)

    async def send_welcome_email(self, email: str, first_name: str) -> bool:
        """Send welcome email to new users."""
        subject = "Welcome to Documind!"
        body = f"""
        <html>
            <body>
                <p>Hi {first_name},</p>
                <br>
                <p>Welcome to Documind! We're excited to have you on board.</p>
                <br>
                <p>You can now start your personalized learning journey with AI-powered guidance.</p>
                <br>
                <p>Thanks,<br>The Documind Team</p>
            </body>
        </html>
        """
        return await self._provider.send_email(email, subject, body)
