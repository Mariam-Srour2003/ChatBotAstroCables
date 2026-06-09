"""
app/email_sender.py — builds and sends the chat transcript as an HTML email.
Uses only <br>, <strong>, <ul>/<li> — no <p> tags — for Gmail compatibility.
"""
import re
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import EMAIL_FROM, EMAIL_PASS, EMAIL_TO, SMTP_HOST, SMTP_PORT


# ── Safe text → HTML (email-client-safe, no <p> tags) ────────────────────────

def _to_email_html(text: str) -> str:
    """
    Convert plain/markdown text to email-safe HTML.
    Rules: escape entities, bold **text**, bullet lists, <br> for newlines.
    No <p> tags — Gmail strips them and shows attribute text as raw characters.
    """
    lines  = text.split("\n")
    out    = []
    in_ul  = False

    for line in lines:
        stripped = line.strip()

        # Blank line → paragraph break
        if stripped == "":
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append("<br>")
            continue

        # Bullet line
        if re.match(r"^[-•*]\s+", stripped):
            if not in_ul:
                out.append(
                    "<ul style='margin:6px 0 6px 16px;padding:0;"
                    "list-style:disc;'>"
                )
                in_ul = True
            item = re.sub(r"^[-•*]\s+", "", stripped)
            out.append(f"<li style='margin:3px 0;'>{_inline(item)}</li>")
            continue

        # Normal line
        if in_ul:
            out.append("</ul>")
            in_ul = False
        out.append(_inline(stripped) + "<br>")

    if in_ul:
        out.append("</ul>")

    # Strip trailing <br> tags
    result = "".join(out)
    result = re.sub(r"(<br>)+$", "", result)
    return result


def _inline(text: str) -> str:
    """Escape HTML and apply inline markdown (bold, italic, code)."""
    s = (text
         .replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;"))
    # Bold **text**
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    # Italic *text*
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    return s


# ── HTML email template ───────────────────────────────────────────────────────

def _build_html(history: list[dict], session_id: str, contact: dict) -> str:
    now      = datetime.now().strftime("%B %d, %Y  &bull;  %I:%M %p")
    count    = len(history) // 2
    rows_html = ""

    for msg in history:
        role    = msg["role"]
        content = _to_email_html(msg["content"])

        if role == "user":
            rows_html += f"""
<tr>
  <td style="padding:0 0 16px 0;">
    <table width="100%" cellspacing="0" cellpadding="0" border="0">
      <tr>
        <td width="40" valign="top">
          <div style="width:34px;height:34px;line-height:34px;border-radius:50%;
                      background:#dbeafe;text-align:center;font-size:16px;">&#128100;</div>
        </td>
        <td style="padding-left:10px;">
          <div style="font-size:10px;font-weight:700;color:#2563eb;
                      text-transform:uppercase;letter-spacing:1px;
                      margin-bottom:5px;">Customer</div>
          <div style="background:#eff6ff;border-left:3px solid #2563eb;
                      border-radius:0 10px 10px 10px;padding:10px 14px;
                      font-size:13.5px;color:#1e293b;line-height:1.7;">
            {content}
          </div>
        </td>
      </tr>
    </table>
  </td>
</tr>"""
        else:
            rows_html += f"""
<tr>
  <td style="padding:0 0 16px 0;">
    <table width="100%" cellspacing="0" cellpadding="0" border="0">
      <tr>
        <td width="40" valign="top">
          <div style="width:34px;height:34px;line-height:34px;border-radius:50%;
                      background:#fff7ed;text-align:center;font-size:16px;">&#9889;</div>
        </td>
        <td style="padding-left:10px;">
          <div style="font-size:10px;font-weight:700;color:#ea580c;
                      text-transform:uppercase;letter-spacing:1px;
                      margin-bottom:5px;">Astro Sales Bot</div>
          <div style="background:#fff7ed;border-left:3px solid #f97316;
                      border-radius:0 10px 10px 10px;padding:10px 14px;
                      font-size:13.5px;color:#1e293b;line-height:1.7;">
            {content}
          </div>
        </td>
      </tr>
    </table>
  </td>
</tr>"""

    plural = "s" if count != 1 else ""

    # ── Contact info block (only when provided) ──────────────────────────────
    contact_block = ""
    if contact:
        rows_ci = ""
        if contact.get("name"):
            rows_ci += (
                f"<tr><td style='font-size:11px;color:#6b7280;padding:1px 0;'>"
                f"&#128100;&nbsp;<strong>Name</strong></td>"
                f"<td style='font-size:12px;color:#1e293b;padding:1px 0 1px 12px;'>"
                f"{_inline(contact['name'])}</td></tr>"
            )
        if contact.get("email"):
            rows_ci += (
                f"<tr><td style='font-size:11px;color:#6b7280;padding:1px 0;'>"
                f"&#9993;&nbsp;<strong>Email</strong></td>"
                f"<td style='font-size:12px;color:#1e293b;padding:1px 0 1px 12px;'>"
                f"{_inline(contact['email'])}</td></tr>"
            )
        if contact.get("phone"):
            rows_ci += (
                f"<tr><td style='font-size:11px;color:#6b7280;padding:1px 0;'>"
                f"&#128222;&nbsp;<strong>Phone</strong></td>"
                f"<td style='font-size:12px;color:#1e293b;padding:1px 0 1px 12px;'>"
                f"{_inline(contact['phone'])}</td></tr>"
            )
        contact_block = f"""
    <!-- Contact card -->
    <tr>
      <td style="background:#f0fdf4;padding:14px 28px;
                 border-bottom:2px solid #bbf7d0;">
        <div style="font-size:10px;font-weight:700;color:#166534;
                    text-transform:uppercase;letter-spacing:1.5px;
                    margin-bottom:8px;">&#128203; Customer Contact Info</div>
        <table cellspacing="0" cellpadding="0" border="0">
          {rows_ci}
        </table>
      </td>
    </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
             font-family:Arial,Helvetica,sans-serif;">

<table width="100%" cellspacing="0" cellpadding="0" border="0"
       style="background:#f1f5f9;padding:30px 16px;">
<tr><td align="center">

  <table width="620" cellspacing="0" cellpadding="0" border="0"
         style="max-width:620px;background:#ffffff;border-radius:16px;
                overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.10);">

    <!-- Header -->
    <tr>
      <td align="center"
          style="background:linear-gradient(135deg,#1B3A6B 0%,#1e52b0 100%);
                 padding:28px 32px;">
        <div style="font-size:30px;line-height:1;">&#9889;</div>
        <div style="font-size:22px;font-weight:800;color:#ffffff;
                    letter-spacing:2px;text-transform:uppercase;
                    margin-top:6px;">Astro Power Cables</div>
        <div style="font-size:11px;color:#93c5fd;letter-spacing:3px;
                    text-transform:uppercase;margin-top:6px;">
          Chat Transcript
        </div>
      </td>
    </tr>

    <!-- Meta bar -->
    <tr>
      <td style="background:#f8faff;padding:13px 28px;
                 border-bottom:1px solid #e2e8f0;">
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
          <tr>
            <td style="font-size:12px;color:#475569;">
              &#128197;&nbsp; <strong>{now}</strong>
            </td>
            <td align="right" style="font-size:11px;color:#94a3b8;">
              {count} exchange{plural}
            </td>
          </tr>
        </table>
      </td>
    </tr>

    {contact_block}

    <!-- Conversation -->
    <tr>
      <td style="padding:24px 28px 8px 28px;">
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
          {rows_html}
        </table>
      </td>
    </tr>

    <!-- Divider -->
    <tr>
      <td style="padding:0 28px;">
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
          <tr><td style="border-top:1px solid #e2e8f0;font-size:0;">&nbsp;</td></tr>
        </table>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td align="center"
          style="background:#1B3A6B;padding:20px 32px;">
        <div style="font-size:14px;font-weight:700;color:#ffffff;
                    margin-bottom:7px;">Astro Power Cables</div>
        <div style="font-size:12px;color:#93c5fd;margin-bottom:4px;">
          &#128222; +961&nbsp;1&nbsp;271&nbsp;471 &nbsp;&nbsp;
          &#128222; +961&nbsp;71&nbsp;271&nbsp;075 &nbsp;&nbsp;
          &#9993; info@astro-lb.com
        </div>
        <div style="font-size:10px;color:#4e6a96;margin-top:8px;">
          ISO 9001:2015 &amp; ISO 14001:2015 Certified &nbsp;&middot;&nbsp;
          Zahle, Lebanon
        </div>
      </td>
    </tr>

  </table>
</td></tr>
</table>

</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def send_transcript(history: list[dict], session_id: str,
                    contact: dict | None = None) -> None:
    """
    Build and send the chat transcript as an HTML email.
    contact: optional dict with keys 'name', 'email', 'phone'.
    Raises on SMTP failure. Silently returns if history is empty.
    """
    if not history:
        return

    contact = contact or {}
    msg = MIMEMultipart("alternative")
    date_str = datetime.now().strftime("%b %d, %Y")
    msg["Subject"] = f"Astro Power Cables — Chat Transcript ({date_str})"
    msg["From"]    = f"Astro Power Cables <{EMAIL_FROM}>"
    msg["To"]      = EMAIL_TO

    msg.attach(MIMEText(_build_html(history, session_id, contact), "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(EMAIL_FROM, EMAIL_PASS)
        smtp.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
