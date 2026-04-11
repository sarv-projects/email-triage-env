"""
Email Triage Environment — Core Implementation.

Simulates a real-world IT / customer-support inbox where an AI agent must
triage incoming emails by classifying category, urgency, and routing action.

Each episode presents a **queue** of emails (5–7 depending on difficulty).
The agent processes them one at a time, receiving shaped rewards per step.

Design highlights
─────────────────
- Multi-step episodes with partial-progress rewards
- 50+ realistic emails across 5 categories and 3 urgency levels
- Thread chains — follow-up emails carry conversational context
- Difficulty scaling: easy (obvious), medium (ambiguous), hard (critical / deceptive)
- Deterministic resets via seed for reproducibility
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from openenv.core.env_server import Environment, State

try:
    from models import EmailAction, EmailObservation, EmailState
except ImportError:
    from ..models import EmailAction, EmailObservation, EmailState


# ═══════════════════════════════════════════════════════════════════════════
# Email Corpus
# ═══════════════════════════════════════════════════════════════════════════
# Each entry:  id, thread_id, thread_position, subject, body, sender,
#              sender_domain, received_at, correct_category, correct_urgency,
#              correct_action, thread_context (for follow-ups)

EASY_EMAILS: List[Dict[str, Any]] = [
    # ── Spam / Phishing (obvious) ─────────────────────────────────────
    {
        "id": "e01", "thread_id": "t-e01", "thread_position": 1,
        "subject": "URGENT: You won $1,000,000 in our lottery!",
        "body": "Congratulations! You have been randomly selected as the winner of our international lottery. Click the link below to claim your prize immediately. No purchase necessary. Act within 24 hours or your winnings expire.",
        "sender": "claims@intl-lotto-prize.com", "sender_domain": "intl-lotto-prize.com",
        "received_at": "2026-04-10T08:12:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e02", "thread_id": "t-e02", "thread_position": 1,
        "subject": "Buy cheap medication online — 90% off!",
        "body": "Best prices on prescription medications shipped worldwide. No prescription needed. Limited-time offer expires tonight. Order now and receive free shipping on all orders over $50.",
        "sender": "deals@pharma-discounts.net", "sender_domain": "pharma-discounts.net",
        "received_at": "2026-04-10T08:30:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e03", "thread_id": "t-e03", "thread_position": 1,
        "subject": "FREE iPhone 16 Pro — You've been selected!",
        "body": "Dear valued customer, you have been exclusively selected to receive a brand new iPhone 16 Pro at no cost. Simply complete our 30-second survey to claim your device. Offer ends today.",
        "sender": "rewards@free-phone-offer.com", "sender_domain": "free-phone-offer.com",
        "received_at": "2026-04-10T09:05:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e04", "thread_id": "t-e04", "thread_position": 1,
        "subject": "Double your Bitcoin investment guaranteed",
        "body": "Join thousands of investors earning 200% returns daily with our AI-powered crypto trading bot. Minimum deposit just $250. Guaranteed returns or your money back. Limited spots remaining.",
        "sender": "invest@crypto-wealth-now.io", "sender_domain": "crypto-wealth-now.io",
        "received_at": "2026-04-10T09:22:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e05", "thread_id": "t-e05", "thread_position": 1,
        "subject": "Your package could not be delivered",
        "body": "We attempted delivery of your package (tracking: XX-9281374) but nobody was available. Click here to reschedule. If not claimed within 48 hours, it will be returned to sender. Delivery fee: $1.99.",
        "sender": "noreply@delivery-notification-center.com", "sender_domain": "delivery-notification-center.com",
        "received_at": "2026-04-10T09:45:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e06", "thread_id": "t-e06", "thread_position": 1,
        "subject": "Lose 30 pounds in 30 days — doctor approved",
        "body": "Revolutionary weight loss supplement used by Hollywood celebrities. No exercise required. Clinical trials show 95% success rate. Order now with our risk-free 60-day money back guarantee.",
        "sender": "results@slimfast-miracle.com", "sender_domain": "slimfast-miracle.com",
        "received_at": "2026-04-10T10:00:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e07", "thread_id": "t-e07", "thread_position": 1,
        "subject": "Advance fee — confidential business proposal",
        "body": "I am Dr. James from the Central Bank of Nigeria. I have a confidential business proposal involving $15.5 million USD. You will receive 30% as commission. Reply with your full name and bank details to proceed.",
        "sender": "dr.james@consultant-cbng.com", "sender_domain": "consultant-cbng.com",
        "received_at": "2026-04-10T10:15:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    # ── Clear legitimate (easy classification) ────────────────────────
    {
        "id": "e08", "thread_id": "t-e08", "thread_position": 1,
        "subject": "Your monthly invoice — April 2026",
        "body": "Hi, your April invoice is ready. Total amount: $149.00. Payment is due by April 25. You can view and pay your invoice in your account dashboard. Thank you for your business.",
        "sender": "billing@acme-saas.com", "sender_domain": "acme-saas.com",
        "received_at": "2026-04-10T10:30:00Z",
        "correct_category": "billing", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e09", "thread_id": "t-e09", "thread_position": 1,
        "subject": "Payment received — thank you",
        "body": "We've received your payment of $299.00 for your annual subscription renewal. Your account has been updated. Next billing date: April 10, 2027. No action is required.",
        "sender": "receipts@cloudhost.io", "sender_domain": "cloudhost.io",
        "received_at": "2026-04-10T10:45:00Z",
        "correct_category": "billing", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e10", "thread_id": "t-e10", "thread_position": 1,
        "subject": "Team standup moved to 10:30 AM",
        "body": "Hi team, just a heads up — tomorrow's standup has been moved from 9:00 AM to 10:30 AM to accommodate the client demo. Same Zoom link. Please update your calendars.",
        "sender": "pm@ourcompany.com", "sender_domain": "ourcompany.com",
        "received_at": "2026-04-10T11:00:00Z",
        "correct_category": "general", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e11", "thread_id": "t-e11", "thread_position": 1,
        "subject": "Password changed successfully",
        "body": "Your password was changed on April 10 at 10:45 AM UTC from a recognized device. If you did not make this change, please contact our security team immediately at security@ourcompany.com.",
        "sender": "noreply@ourcompany.com", "sender_domain": "ourcompany.com",
        "received_at": "2026-04-10T11:15:00Z",
        "correct_category": "security", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e12", "thread_id": "t-e12", "thread_position": 1,
        "subject": "Welcome to Acme Platform",
        "body": "Welcome aboard! Your Acme Platform account is now active. Here's what you can do next: complete your profile, explore our API docs, and join our Slack community. Happy building!",
        "sender": "onboarding@acme-saas.com", "sender_domain": "acme-saas.com",
        "received_at": "2026-04-10T11:30:00Z",
        "correct_category": "general", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e13", "thread_id": "t-e13", "thread_position": 1,
        "subject": "Scheduled maintenance — April 15",
        "body": "We will perform scheduled maintenance on April 15 from 2:00 AM to 4:00 AM UTC. During this window, the API may be intermittently unavailable. No data will be affected. We apologize for any inconvenience.",
        "sender": "status@cloudhost.io", "sender_domain": "cloudhost.io",
        "received_at": "2026-04-10T11:45:00Z",
        "correct_category": "technical", "correct_urgency": "low", "correct_action": "archive",
        "thread_context": "",
    },
    {
        "id": "e14", "thread_id": "t-e14", "thread_position": 1,
        "subject": "Dating matches waiting for you!",
        "body": "You have 5 new matches on LoveConnect! Don't keep them waiting. Upgrade to Premium for unlimited messaging and see who liked your profile. Special offer: 50% off this week only.",
        "sender": "matches@loveconnect-app.com", "sender_domain": "loveconnect-app.com",
        "received_at": "2026-04-10T12:00:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "e15", "thread_id": "t-e15", "thread_position": 1,
        "subject": "Quick question about API rate limits",
        "body": "Hi support, I was reading the docs and wanted to confirm: is the rate limit 1000 requests per minute on the Pro plan? Also, do WebSocket connections count against this limit? Thanks!",
        "sender": "dev@startup.io", "sender_domain": "startup.io",
        "received_at": "2026-04-10T12:15:00Z",
        "correct_category": "technical", "correct_urgency": "low", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "e16", "thread_id": "t-e16", "thread_position": 1,
        "subject": "Exclusive offer — business loan pre-approved",
        "body": "You've been pre-approved for a business loan up to $500,000 with rates as low as 2.9% APR. No collateral required. Apply in minutes. This offer expires in 72 hours.",
        "sender": "offers@quickbiz-loans.com", "sender_domain": "quickbiz-loans.com",
        "received_at": "2026-04-10T12:30:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
]

MEDIUM_EMAILS: List[Dict[str, Any]] = [
    # ── Billing disputes & questions ──────────────────────────────────
    {
        "id": "m01", "thread_id": "t-m01", "thread_position": 1,
        "subject": "I think I was double-charged last month",
        "body": "Hi, I noticed two charges of $149.00 on my credit card statement from March — one on March 1st and another on March 15th. I believe I should only have been charged once. Can you look into this and process a refund if applicable?",
        "sender": "client@megacorp.com", "sender_domain": "megacorp.com",
        "received_at": "2026-04-10T13:00:00Z",
        "correct_category": "billing", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "m02", "thread_id": "t-m01", "thread_position": 2,
        "subject": "Re: I think I was double-charged last month",
        "body": "Following up on my earlier email — I still haven't received a response about the duplicate charge. It has now been three business days. Please prioritize this. I will dispute the charge with my bank if not resolved by end of week.",
        "sender": "client@megacorp.com", "sender_domain": "megacorp.com",
        "received_at": "2026-04-10T15:00:00Z",
        "correct_category": "billing", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "Prior message: Customer reported a potential duplicate charge of $149.00 in March billing. No response sent yet.",
    },
    {
        "id": "m03", "thread_id": "t-m03", "thread_position": 1,
        "subject": "Want to cancel my subscription",
        "body": "Hi, I'd like to cancel my Pro subscription effective at the end of this billing cycle. Can you confirm the cancellation and let me know if there's a prorated refund for the remaining days? I've been happy with the service but our budget has changed.",
        "sender": "user@smallbiz.co", "sender_domain": "smallbiz.co",
        "received_at": "2026-04-10T13:15:00Z",
        "correct_category": "billing", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    # ── Technical support (moderate) ──────────────────────────────────
    {
        "id": "m04", "thread_id": "t-m04", "thread_position": 1,
        "subject": "API returning 502 errors intermittently",
        "body": "We've been experiencing intermittent 502 Bad Gateway errors on the /v2/data endpoint since about 11 AM today. It happens roughly every 5th request. Our service handles about 200 req/min through this endpoint. Retry logic helps but it's slowing us down.",
        "sender": "backend@partner-tech.com", "sender_domain": "partner-tech.com",
        "received_at": "2026-04-10T13:30:00Z",
        "correct_category": "technical", "correct_urgency": "medium", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "m05", "thread_id": "t-m05", "thread_position": 1,
        "subject": "Feature request: batch export to CSV",
        "body": "Our analytics team frequently needs to export large datasets (50K+ rows) from the dashboard. Currently it's limited to 1000 rows per export. Would it be possible to add a batch CSV export feature? This would save us hours of manual work each week.",
        "sender": "analyst@datadrive.com", "sender_domain": "datadrive.com",
        "received_at": "2026-04-10T13:45:00Z",
        "correct_category": "technical", "correct_urgency": "low", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "m06", "thread_id": "t-m06", "thread_position": 1,
        "subject": "Webhook deliveries failing since upgrade",
        "body": "After upgrading to v3.2 of your SDK last week, our webhook endpoint is no longer receiving delivery confirmations. The v3.1 setup was working fine. We haven't changed anything on our end. Can you check if something changed in the webhook payload format?",
        "sender": "devops@ecomstore.com", "sender_domain": "ecomstore.com",
        "received_at": "2026-04-10T14:00:00Z",
        "correct_category": "technical", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    # ── General / ambiguous ───────────────────────────────────────────
    {
        "id": "m07", "thread_id": "t-m07", "thread_position": 1,
        "subject": "Partnership inquiry — integration opportunity",
        "body": "Hi, I'm the Head of Partnerships at DataSync Inc. We're building an integration marketplace and would love to feature your API. Could we schedule a 30-minute call to discuss a potential partnership? We currently serve 500+ enterprise clients.",
        "sender": "partnerships@datasync.io", "sender_domain": "datasync.io",
        "received_at": "2026-04-10T14:15:00Z",
        "correct_category": "general", "correct_urgency": "low", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "m08", "thread_id": "t-m08", "thread_position": 1,
        "subject": "Feedback on the new dashboard redesign",
        "body": "I just wanted to share that the new dashboard redesign is really frustrating. The navigation is confusing, charts take forever to load, and I can't find the export button anymore. Several people on my team feel the same way. Please consider reverting the changes.",
        "sender": "manager@retail-chain.com", "sender_domain": "retail-chain.com",
        "received_at": "2026-04-10T14:30:00Z",
        "correct_category": "technical", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "m09", "thread_id": "t-m09", "thread_position": 1,
        "subject": "Question about enterprise pricing",
        "body": "We're evaluating your platform for our 200-person engineering team. Can you share details on enterprise pricing, volume discounts, and whether you offer annual billing? Also, do you provide dedicated support for enterprise accounts?",
        "sender": "procurement@fortune500.com", "sender_domain": "fortune500.com",
        "received_at": "2026-04-10T14:45:00Z",
        "correct_category": "billing", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    # ── Security (moderate) ───────────────────────────────────────────
    {
        "id": "m10", "thread_id": "t-m10", "thread_position": 1,
        "subject": "Unusual activity on team account",
        "body": "I noticed a login to our team account from an IP address in a country where we don't have employees (IP: 185.220.xx.xx, location: Romania). The session lasted 4 minutes and no changes were made that I can see. Should I be concerned? I've changed my password already.",
        "sender": "admin@consulting-group.com", "sender_domain": "consulting-group.com",
        "received_at": "2026-04-10T15:00:00Z",
        "correct_category": "security", "correct_urgency": "medium", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "m11", "thread_id": "t-m11", "thread_position": 1,
        "subject": "Can you add SSO / SAML support?",
        "body": "Our IT security policy requires SSO for all third-party tools. We currently can't roll out your platform company-wide without SAML 2.0 integration. Is this on your roadmap? If so, when can we expect it? This is a blocker for our deployment.",
        "sender": "it-security@enterprise.org", "sender_domain": "enterprise.org",
        "received_at": "2026-04-10T15:15:00Z",
        "correct_category": "security", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    # ── Subtle spam (medium difficulty) ───────────────────────────────
    {
        "id": "m12", "thread_id": "t-m12", "thread_position": 1,
        "subject": "Invitation: Exclusive SaaS Growth Summit",
        "body": "You're invited to the SaaS Growth Summit 2026 — an exclusive, invite-only event for SaaS founders and executives. Early bird tickets are $1,999 (regular $3,999). Speakers include leaders from top tech companies. Register now to secure your spot.",
        "sender": "events@saas-summit-global.com", "sender_domain": "saas-summit-global.com",
        "received_at": "2026-04-10T15:30:00Z",
        "correct_category": "spam", "correct_urgency": "low", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "m13", "thread_id": "t-m13", "thread_position": 1,
        "subject": "Important update to your service agreement",
        "body": "We're writing to inform you of changes to our Terms of Service effective May 1, 2026. Key changes include updated data processing terms and a revised SLA for Enterprise accounts. Please review the updated terms at our legal page. Continued use constitutes acceptance.",
        "sender": "legal@cloudhost.io", "sender_domain": "cloudhost.io",
        "received_at": "2026-04-10T15:45:00Z",
        "correct_category": "general", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
    # ── Thread follow-up ──────────────────────────────────────────────
    {
        "id": "m14", "thread_id": "t-m04", "thread_position": 2,
        "subject": "Re: API returning 502 errors intermittently",
        "body": "Update: The 502 errors have increased in frequency over the last 2 hours. We're now seeing failures on roughly 30% of requests. Our uptime monitor is flagging alerts. Can someone from your infrastructure team look at this urgently?",
        "sender": "backend@partner-tech.com", "sender_domain": "partner-tech.com",
        "received_at": "2026-04-10T16:00:00Z",
        "correct_category": "technical", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "Prior message: Customer reported intermittent 502 errors on /v2/data endpoint (~20% failure rate, 200 req/min). Issue classified as medium urgency.",
    },
    {
        "id": "m15", "thread_id": "t-m15", "thread_position": 1,
        "subject": "Please update our billing contact",
        "body": "Hi, our finance team has changed. Please update the billing contact from john.doe@megacorp.com to jane.smith@megacorp.com for all future invoices. Also, can you resend the last three invoices to the new address? Thanks.",
        "sender": "admin@megacorp.com", "sender_domain": "megacorp.com",
        "received_at": "2026-04-10T16:15:00Z",
        "correct_category": "billing", "correct_urgency": "low", "correct_action": "respond",
        "thread_context": "",
    },
    {
        "id": "m16", "thread_id": "t-m16", "thread_position": 1,
        "subject": "Performance degradation on reporting module",
        "body": "The weekly report generation that used to take 2 minutes now takes over 15 minutes. This started after the March 28 platform update. We rely on these reports for our Monday morning executive briefing. Not critical yet but getting worse.",
        "sender": "ops@logistics-co.com", "sender_domain": "logistics-co.com",
        "received_at": "2026-04-10T16:30:00Z",
        "correct_category": "technical", "correct_urgency": "medium", "correct_action": "respond",
        "thread_context": "",
    },
]

HARD_EMAILS: List[Dict[str, Any]] = [
    # ── Critical billing / financial ──────────────────────────────────
    {
        "id": "h01", "thread_id": "t-h01", "thread_position": 1,
        "subject": "CRITICAL: Payment processing failure — production impact",
        "body": "Our payment integration has been returning 500 errors for the last 45 minutes. We process approximately $80K in transactions per hour and are currently unable to accept any payments. Our checkout page is completely non-functional. This is a P0 incident for us.",
        "sender": "cto@ecomm-giant.com", "sender_domain": "ecomm-giant.com",
        "received_at": "2026-04-10T17:00:00Z",
        "correct_category": "technical", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
    {
        "id": "h02", "thread_id": "t-h02", "thread_position": 1,
        "subject": "SECURITY ALERT: Unauthorized API key usage detected",
        "body": "Our monitoring flagged that API key sk-prod-****7f2a is being used from IP addresses not in our allowlist. We've detected 12,000+ requests in the last hour from three different countries. The key has production-level permissions including write access to customer data. We've rotated the key on our end but need to know if any data was accessed.",
        "sender": "security@fintech-corp.com", "sender_domain": "fintech-corp.com",
        "received_at": "2026-04-10T17:15:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
    {
        "id": "h03", "thread_id": "t-h03", "thread_position": 1,
        "subject": "GDPR Data Subject Access Request — 30-day deadline",
        "body": "Pursuant to Article 15 of the General Data Protection Regulation (EU 2016/679), I am exercising my right of access. Please provide all personal data you hold about me within 30 calendar days. This includes data in backups, logs, analytics platforms, and third-party processors. Failure to comply may result in a complaint to the supervisory authority.",
        "sender": "privacy@lawfirm-mueller.de", "sender_domain": "lawfirm-mueller.de",
        "received_at": "2026-04-10T17:30:00Z",
        "correct_category": "general", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    # ── Executive escalation ──────────────────────────────────────────
    {
        "id": "h04", "thread_id": "t-h04", "thread_position": 1,
        "subject": "Disappointed with service — considering alternatives",
        "body": "I'm the VP of Engineering at TechForward Inc (one of your largest enterprise accounts, $240K ARR). Over the past month we've experienced three major outages, unresponsive support, and broken webhook integrations. I'm meeting with your competitor next week. I'd appreciate a call with your leadership to discuss our concerns before we make a decision.",
        "sender": "vp-eng@techforward.com", "sender_domain": "techforward.com",
        "received_at": "2026-04-10T17:45:00Z",
        "correct_category": "general", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
    {
        "id": "h05", "thread_id": "t-h05", "thread_position": 1,
        "subject": "Invoice dispute — $47,500 discrepancy",
        "body": "Our accounts payable team has identified a $47,500 discrepancy between the contracted rate and the invoice we received for Q1 2026. The contract specifies $12,500/month for the Enterprise plan but we were billed $28,250 for March alone. Please provide a corrected invoice and a credit for the overpayment within 5 business days.",
        "sender": "finance@global-retail.com", "sender_domain": "global-retail.com",
        "received_at": "2026-04-10T18:00:00Z",
        "correct_category": "billing", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    # ── Sophisticated phishing (looks real) ───────────────────────────
    {
        "id": "h06", "thread_id": "t-h06", "thread_position": 1,
        "subject": "Action required: Verify your account to avoid suspension",
        "body": "We've detected unusual activity on your account and have temporarily limited certain features. To restore full access, please verify your identity by clicking the secure link below. If you don't verify within 24 hours, your account will be permanently suspended. This is an automated security measure.",
        "sender": "security-team@cl0udhost.io", "sender_domain": "cl0udhost.io",
        "received_at": "2026-04-10T18:15:00Z",
        "correct_category": "spam", "correct_urgency": "medium", "correct_action": "mark_spam",
        "thread_context": "",
    },
    {
        "id": "h07", "thread_id": "t-h07", "thread_position": 1,
        "subject": "Urgent: Update your payment method",
        "body": "Your credit card ending in 4242 will expire soon. To avoid service interruption, please update your payment details in your account settings. If your service is interrupted, there may be data loss. Click here to update your billing information securely.",
        "sender": "billing@acme-saas-support.com", "sender_domain": "acme-saas-support.com",
        "received_at": "2026-04-10T18:30:00Z",
        "correct_category": "spam", "correct_urgency": "medium", "correct_action": "mark_spam",
        "thread_context": "",
    },
    # ── Production / incident ─────────────────────────────────────────
    {
        "id": "h08", "thread_id": "t-h01", "thread_position": 2,
        "subject": "Re: CRITICAL: Payment processing failure — production impact",
        "body": "It's now been 2 hours. We've lost over $160K in potential revenue. Our CEO is asking for an incident report. We need an ETA on the fix immediately and a direct line to your on-call engineer. If this isn't resolved in the next hour, we're activating our backup provider and will be discussing SLA credits.",
        "sender": "cto@ecomm-giant.com", "sender_domain": "ecomm-giant.com",
        "received_at": "2026-04-10T19:00:00Z",
        "correct_category": "technical", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "Prior message: Payment integration returning 500 errors for 45 min. Customer processing $80K/hr. Checkout completely down. Classified as P0.",
    },
    {
        "id": "h09", "thread_id": "t-h09", "thread_position": 1,
        "subject": "Data breach notification — regulatory obligation",
        "body": "During a routine security audit, we discovered that a misconfigured S3 bucket exposed customer PII (names, emails, phone numbers) for approximately 48 hours between April 5-7. We've secured the bucket and are assessing the scope. Under GDPR and CCPA, we're obligated to notify affected users within 72 hours. We need your data processing records urgently.",
        "sender": "dpo@partner-finserv.com", "sender_domain": "partner-finserv.com",
        "received_at": "2026-04-10T19:15:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
    # ── Ambiguous / nuanced ───────────────────────────────────────────
    {
        "id": "h10", "thread_id": "t-h10", "thread_position": 1,
        "subject": "Compliance audit — documentation request",
        "body": "As part of our SOC 2 Type II audit, our auditors require documentation of your data handling practices, encryption standards, backup policies, and incident response procedures. We need this within 10 business days or our compliance certification is at risk. Please assign someone from your security team to coordinate.",
        "sender": "compliance@healthtech-inc.com", "sender_domain": "healthtech-inc.com",
        "received_at": "2026-04-10T19:30:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "h11", "thread_id": "t-h02", "thread_position": 2,
        "subject": "Re: SECURITY ALERT: Unauthorized API key usage detected",
        "body": "We've completed our internal investigation. The compromised key was used to read 14,000 customer records over a 3-hour window. We've isolated the affected data and engaged our legal counsel. We need your server-side access logs for the affected timeframe (April 9, 14:00-17:00 UTC) within 24 hours for our incident report.",
        "sender": "security@fintech-corp.com", "sender_domain": "fintech-corp.com",
        "received_at": "2026-04-10T19:45:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "Prior message: API key sk-prod-****7f2a used from unauthorized IPs (12K+ requests from 3 countries). Key rotated. Customer asking about data access scope.",
    },
    {
        "id": "h12", "thread_id": "t-h12", "thread_position": 1,
        "subject": "Legal notice: Copyright infringement claim",
        "body": "We represent VisualStock Inc. It has come to our attention that your platform is hosting user-uploaded content that infringes on our client's copyrighted material (Reference: VS-2026-0419). Under the DMCA, we demand immediate removal of the infringing content. Failure to comply within 48 hours will result in formal legal proceedings.",
        "sender": "dmca@legalnotice-firm.com", "sender_domain": "legalnotice-firm.com",
        "received_at": "2026-04-10T20:00:00Z",
        "correct_category": "general", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "h13", "thread_id": "t-h13", "thread_position": 1,
        "subject": "Account suspension warning — ToS violation",
        "body": "We've identified activity on your account that violates Section 4.2 of our Terms of Service (automated scraping). Your account will be suspended in 48 hours unless you cease the violating activity and respond with an explanation. If you believe this is an error, reply with details of your usage patterns.",
        "sender": "trust-safety@cloudhost.io", "sender_domain": "cloudhost.io",
        "received_at": "2026-04-10T20:15:00Z",
        "correct_category": "general", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "h14", "thread_id": "t-h14", "thread_position": 1,
        "subject": "Vulnerability disclosure: XSS in customer portal",
        "body": "I'm a security researcher. I've discovered a stored XSS vulnerability in your customer portal (/dashboard/settings/profile). An attacker could inject arbitrary JavaScript through the 'company name' field, stealing session cookies of any user viewing the profile. I'm following responsible disclosure practices and will publish after 90 days. Please assign a CVE.",
        "sender": "researcher@bugbounty-hq.com", "sender_domain": "bugbounty-hq.com",
        "received_at": "2026-04-10T20:30:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
    {
        "id": "h15", "thread_id": "t-h03", "thread_position": 2,
        "subject": "Re: GDPR Data Subject Access Request — final reminder",
        "body": "This is a follow-up to my DSAR sent 25 days ago. You have 5 calendar days remaining to comply. I have not received any acknowledgment or data. Failure to respond will result in an immediate complaint to the Irish Data Protection Commission. Our legal team is prepared to pursue enforcement.",
        "sender": "privacy@lawfirm-mueller.de", "sender_domain": "lawfirm-mueller.de",
        "received_at": "2026-04-10T20:45:00Z",
        "correct_category": "general", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "Prior message: Formal GDPR Article 15 Data Subject Access Request demanding all personal data within 30 days (backups, logs, analytics, 3rd-party processors). Legal action threatened.",
    },
    {
        "id": "h16", "thread_id": "t-h16", "thread_position": 1,
        "subject": "SLA breach — requesting credits per contract",
        "body": "Per our Enterprise agreement (contract #ENT-2025-0892), you guarantee 99.95% uptime. Your status page shows 4 hours of downtime this month (99.72%). This triggers the SLA credit clause (Section 7.3). We're requesting $18,750 in service credits as specified. Please process and confirm.",
        "sender": "legal@enterprise-client.com", "sender_domain": "enterprise-client.com",
        "received_at": "2026-04-10T21:00:00Z",
        "correct_category": "billing", "correct_urgency": "high", "correct_action": "escalate",
        "thread_context": "",
    },
    {
        "id": "h17", "thread_id": "t-h17", "thread_position": 1,
        "subject": "Whistleblower report: Internal data misuse",
        "body": "I'm an employee at a company that uses your platform. I have evidence that a colleague has been exporting customer data from your system and selling it to third parties. This involves approximately 50,000 records. I want to report this through proper channels. Can you connect me with your data protection officer?",
        "sender": "anonymous9847@protonmail.com", "sender_domain": "protonmail.com",
        "received_at": "2026-04-10T21:15:00Z",
        "correct_category": "security", "correct_urgency": "high", "correct_action": "escalate_urgent",
        "thread_context": "",
    },
]

# ── Valid values for normalization ────────────────────────────────────────
VALID_CATEGORIES = {"spam", "billing", "technical", "general", "security"}
VALID_URGENCY = {"low", "medium", "high"}
VALID_ACTIONS = {"mark_spam", "archive", "respond", "escalate", "escalate_urgent"}

# ── Near-miss pairs for partial credit ────────────────────────────────────
CATEGORY_NEAR_MISS = {
    frozenset({"billing", "general"}),
    frozenset({"technical", "security"}),
    frozenset({"spam", "general"}),
}

ACTION_NEAR_MISS = {
    frozenset({"archive", "respond"}),
    frozenset({"escalate", "escalate_urgent"}),
    frozenset({"mark_spam", "archive"}),
}

# ── Queue sizes per difficulty ────────────────────────────────────────────
QUEUE_SIZE = {"easy": 5, "medium": 6, "hard": 7}


# ═══════════════════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════════════════

class EmailEnvironment(Environment):
    """Multi-step email triage environment.

    Each episode presents a queue of emails. The agent classifies one email
    per step and receives a shaped reward in [0, 1]. The episode ends once
    all emails have been processed.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = EmailState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name="easy",
        )
        self._queue: List[Dict[str, Any]] = []
        self._queue_index: int = 0
        self._rewards: List[float] = []
        self._thread_decisions: Dict[str, str] = {}  # thread_id → last category

    # ── reset ─────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        """Reset the environment and present the first email in the queue.

        Args:
            seed: Random seed for deterministic episode generation.
            episode_id: Optional episode identifier.
            **kwargs:
                task_name: "easy" | "medium" | "hard" (default: "easy")
        """
        task_name = str(kwargs.get("task_name", "easy")).lower()
        if task_name not in QUEUE_SIZE:
            task_name = "easy"

        pool = self._get_pool(task_name)
        rng = random.Random(seed)
        n = min(QUEUE_SIZE[task_name], len(pool))
        self._queue = rng.sample(pool, n)
        self._queue_index = 0
        self._rewards = []
        self._thread_decisions = {}

        self._state = EmailState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            emails_total=n,
            emails_processed=0,
            current_thread_id=self._queue[0]["thread_id"] if self._queue else "",
            cumulative_reward=0.0,
        )

        return self._make_observation(done=False, reward=0.0)

    # ── step ──────────────────────────────────────────────────────────
    def step(
        self,
        action: EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        """Score the agent's classification for the current email and advance."""
        _ = timeout_s, kwargs

        if not self._queue or self._queue_index >= len(self._queue):
            raise RuntimeError("Environment must be reset before step().")

        self._state.step_count += 1
        email = self._queue[self._queue_index]
        reward = self._score(action, email)
        self._rewards.append(reward)

        # Track thread decisions for consistency
        self._thread_decisions[email["thread_id"]] = action.category

        # Advance queue
        self._queue_index += 1
        done = self._queue_index >= len(self._queue)

        self._state.emails_processed = self._queue_index
        self._state.cumulative_reward = sum(self._rewards)
        if not done:
            self._state.current_thread_id = self._queue[self._queue_index]["thread_id"]

        return self._make_observation(done=done, reward=round(reward, 2))

    # ── state ─────────────────────────────────────────────────────────
    @property
    def state(self) -> EmailState:
        """Return the current environment state."""
        return self._state

    # ── internal helpers ──────────────────────────────────────────────
    @staticmethod
    def _get_pool(task_name: str) -> List[Dict[str, Any]]:
        pools = {
            "easy": EASY_EMAILS,
            "medium": MEDIUM_EMAILS,
            "hard": HARD_EMAILS,
        }
        return pools.get(task_name, EASY_EMAILS)

    def _make_observation(self, *, done: bool, reward: float) -> EmailObservation:
        """Build an observation for the current queue position."""
        if done or self._queue_index >= len(self._queue):
            return EmailObservation(
                subject="",
                body="[All emails in the queue have been processed.]",
                sender="",
                sender_domain="",
                received_at="",
                thread_id="",
                thread_position=0,
                thread_context="",
                emails_remaining=0,
                emails_processed=self._queue_index,
                queue_summary=f"{len(self._queue)} emails processed. Episode complete.",
                done=True,
                reward=reward,
            )

        email = self._queue[self._queue_index]
        remaining = len(self._queue) - self._queue_index - 1
        high_remaining = sum(
            1 for e in self._queue[self._queue_index + 1:]
            if e["correct_urgency"] == "high"
        )
        summary_parts = [f"{remaining + 1} email(s) remaining"]
        if high_remaining:
            summary_parts.append(f"{high_remaining} high priority")

        return EmailObservation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            sender_domain=email["sender_domain"],
            received_at=email["received_at"],
            thread_id=email["thread_id"],
            thread_position=email["thread_position"],
            thread_context=email.get("thread_context", ""),
            emails_remaining=remaining,
            emails_processed=self._queue_index,
            queue_summary=", ".join(summary_parts),
            done=done,
            reward=reward,
        )

    def _score(self, action: EmailAction, email: Dict[str, Any]) -> float:
        """Compute a shaped reward in [0, 1] for a single classification."""
        score = 0.0

        # ── Category (max 0.40) ───────────────────────────────────────
        cat_pred = action.category.strip().lower()
        cat_true = email["correct_category"]
        if cat_pred == cat_true:
            score += 0.40
        elif frozenset({cat_pred, cat_true}) in CATEGORY_NEAR_MISS:
            score += 0.15

        # ── Urgency (max 0.25) ────────────────────────────────────────
        urg_pred = action.urgency.strip().lower()
        urg_true = email["correct_urgency"]
        if urg_pred == urg_true:
            score += 0.25
        elif abs(self._urgency_ord(urg_pred) - self._urgency_ord(urg_true)) == 1:
            score += 0.10

        # ── Action (max 0.25) ─────────────────────────────────────────
        act_pred = action.action.strip().lower()
        act_true = email["correct_action"]
        if act_pred == act_true:
            score += 0.25
        elif frozenset({act_pred, act_true}) in ACTION_NEAR_MISS:
            score += 0.10

        # ── Thread consistency bonus (max 0.10) ──────────────────────
        thread_id = email["thread_id"]
        if thread_id in self._thread_decisions:
            prev_cat = self._thread_decisions[thread_id]
            if cat_pred == prev_cat and cat_pred == cat_true:
                score += 0.10
            elif cat_pred == prev_cat:
                score += 0.05

        # ── Penalties ─────────────────────────────────────────────────
        # Marking a legitimate email as spam
        if act_pred == "mark_spam" and cat_true != "spam":
            score -= 0.20

        # Ignoring/archiving a high-urgency email
        if act_pred in ("archive", "mark_spam") and urg_true == "high":
            score -= 0.15

        return max(0.0, min(1.0, round(score, 2)))

    @staticmethod
    def _urgency_ord(u: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(u, 1)


# ── Standalone test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    env = EmailEnvironment()
    for task in ("easy", "medium", "hard"):
        obs = env.reset(seed=42, task_name=task)
        print(f"\n{'='*60}")
        print(f"Task: {task} | Queue: {env.state.emails_total} emails")
        total_reward = 0.0
        step_n = 0
        while not obs.done:
            step_n += 1
            # Perfect agent: use ground truth
            email = env._queue[env._queue_index]
            action = EmailAction(
                category=email["correct_category"],
                urgency=email["correct_urgency"],
                action=email["correct_action"],
            )
            obs = env.step(action)
            total_reward += obs.reward
            print(f"  Step {step_n}: reward={obs.reward:.2f} done={obs.done}")
        avg = total_reward / step_n if step_n else 0
        print(f"  => Average reward: {avg:.3f}")
