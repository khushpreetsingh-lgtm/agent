# Gmail Agent — Manual Test Plan

> **Before you start:**
> - Replace `alice@example.com`, `bob@company.com` with real email addresses in your contacts
> - Replace `"project update"`, `"invoice"` etc. with real subject lines from your inbox
> - Each test is independent — start a **fresh message** for each one
> - `user_google_email` is auto-injected — you should NEVER be asked for your own email address
> - A checkmark column is left for you to mark pass/fail

---

## SECTION 1 — READ / SEARCH (should be fast, no workflow)

| # | Type in chat | Expected response |
|---|---|---|
| 1.1 | `show my unread emails` | List of unread messages with sender, subject, date |
| 1.2 | `check my inbox` | Recent inbox messages listed |
| 1.3 | `any new emails?` | Unread count or list |
| 1.4 | `show emails from alice@example.com` | Emails from that sender |
| 1.5 | `show emails from Alice` | Emails from name match |
| 1.6 | `find emails about project update` | Emails with that subject/body keyword |
| 1.7 | `search for emails with subject invoice` | Emails matching subject |
| 1.8 | `show starred emails` | Starred messages |
| 1.9 | `show sent emails` | Sent mail |
| 1.10 | `show emails I sent today` | Today's sent messages |
| 1.11 | `show emails from last week` | Messages from past 7 days |
| 1.12 | `how many unread emails do I have` | Count of unread messages |
| 1.13 | `show emails with attachments` | Messages with files attached |
| 1.14 | `show important emails` | Emails labelled Important |
| 1.15 | `show emails in spam` | Spam folder contents |

**Empty state tests (expect a clean friendly sentence, NOT raw JSON or "No results"):**

| # | Type in chat | Expected |
|---|---|---|
| 1.16 | `show starred emails` | If none: "You don't have any starred emails right now." |
| 1.17 | `show emails from nobody@fake.com` | If none: "No emails found from that address." |
| 1.18 | `show emails with subject xyz_nonexistent` | If none: "No emails found matching that subject." |

---

## SECTION 2 — READ A SPECIFIC EMAIL

| # | Type in chat | Expected flow |
|---|---|---|
| 2.1 | `read my latest unread email` | Searches unread → reads newest → shows full content |
| 2.2 | `read the email from Alice about the project` | Searches → shows content |
| 2.3 | `show me the email about invoice` | Searches → if 1 result, reads it. If multiple, asks which one |
| 2.4 | `open latest email from bob@company.com` | Searches → reads → shows body |
| 2.5 | `what does the email from Alice say` | Reads and summarises |

**Multiple results — should ask which one:**

| # | Type in chat | Expected |
|---|---|---|
| 2.6 | `read emails from Alice` (many exist) | Lists results → asks which to open → reads selected |

---

## SECTION 3 — SEND EMAIL

**Full info given — should NOT ask for anything:**

| # | Type in chat | Expected flow |
|---|---|---|
| 3.1 | `send an email to alice@example.com with subject Hello and body Hi there` | Sends directly, confirms |
| 3.2 | `email bob@company.com: subject Meeting, body Let's meet at 3pm` | Sends directly |
| 3.3 | `send alice@example.com the message "Can you review the doc?"` | Sends, confirms with subject auto-generated or asks |

**Partial info — should ask only for what's missing:**

| # | Type in chat | Expected flow |
|---|---|---|
| 3.4 | `send an email to alice@example.com` | Asks subject → asks body → sends → confirms |
| 3.5 | `send an email` | Asks to → asks subject → asks body → sends |
| 3.6 | `compose an email` | Same as 3.5 |
| 3.7 | `write an email to alice@example.com about the project` | Has to + subject hint → asks body → sends |

**Should NOT re-ask for info already given:**

| # | Type in chat | Expected |
|---|---|---|
| 3.8 | `send alice@example.com subject "Follow-up" body "Just checking in"` | Sends immediately — NO questions |
| 3.9 | `email alice: Hi there, thanks for your help` | Asks for email address (Alice has no email in message), sends |

---

## SECTION 4 — REPLY TO EMAIL

| # | Type in chat | Expected flow |
|---|---|---|
| 4.1 | `reply to the latest email from Alice` | Finds email → asks what to reply → sends in thread |
| 4.2 | `reply to Alice's email saying "Got it, thanks"` | Finds email → sends reply directly |
| 4.3 | `reply to the email about the invoice` | Searches → reads → asks reply content → sends in thread |
| 4.4 | `reply all to Alice's email` | Searches → asks content → sends reply-all |

---

## SECTION 5 — FORWARD EMAIL

| # | Type in chat | Expected flow |
|---|---|---|
| 5.1 | `forward the latest email from Alice to bob@company.com` | Finds → forwards → confirms |
| 5.2 | `forward the invoice email to accounting@company.com` | Finds → forwards |
| 5.3 | `forward Alice's last email to Bob with a note "FYI"` | Finds → composes forward with note → sends |

---

## SECTION 6 — DRAFT EMAIL

| # | Type in chat | Expected flow |
|---|---|---|
| 6.1 | `draft an email to alice@example.com subject "Proposal" body "Please review"` | Creates draft, confirms (not sent) |
| 6.2 | `save a draft to alice@example.com` | Asks subject → asks body → saves draft |
| 6.3 | `draft email to the team about the release` | Asks recipient → asks body → saves draft |

---

## SECTION 7 — LABEL MANAGEMENT

| # | Type in chat | Expected flow |
|---|---|---|
| 7.1 | `mark the email from Alice as read` | Finds → marks read → confirms |
| 7.2 | `mark all emails from Alice as read` | Searches → marks all read → confirms count |
| 7.3 | `star the email about the invoice` | Finds → adds star → confirms |
| 7.4 | `archive the latest email from Alice` | Finds → archives → confirms |
| 7.5 | `move the email from Alice to spam` | Finds → marks spam → confirms |
| 7.6 | `unstar the email about the project` | Finds → removes star → confirms |
| 7.7 | `mark email from Bob as important` | Finds → marks important → confirms |

---

## SECTION 8 — THREAD / CONVERSATION VIEW

| # | Type in chat | Expected flow |
|---|---|---|
| 8.1 | `show the full thread of Alice's email` | Searches → gets thread content → shows all messages |
| 8.2 | `show the conversation about the project update` | Searches → shows thread |
| 8.3 | `how many replies does the invoice email thread have` | Gets thread → counts → answers |

---

## SECTION 9 — BULK OPERATIONS

| # | Type in chat | Expected flow |
|---|---|---|
| 9.1 | `mark all emails from newsletter@company.com as read` | Searches → batch mark read → confirms count |
| 9.2 | `archive all emails from Alice older than a week` | Searches → confirms count → archives |
| 9.3 | `delete all emails from newsletter@company.com` | Searches → asks confirmation → deletes |

---

## SECTION 10 — EDGE CASES

### Should NOT send an email

| # | Type in chat | Expected |
|---|---|---|
| 10.1 | `how many unread emails do I have` | Returns count — does NOT call send_gmail_message |
| 10.2 | `who sent me emails today` | Lists senders — does NOT send |
| 10.3 | `what is Alice's email?` | Searches contacts/emails — does NOT send |

### Should NOT ask for user's own email

| # | Type in chat | Expected |
|---|---|---|
| 10.4 | `send an email to alice@example.com` | Should NEVER ask "What is your email address?" — it's auto-injected |
| 10.5 | `check my inbox` | Should NEVER ask for your Gmail address |
| 10.6 | `show my sent emails` | Should NEVER ask for your email |

### Missing field — should ask exactly what's missing

| # | Type in chat | Expected |
|---|---|---|
| 10.7 | `send an email` | Asks: who to? then subject? then body? — nothing else |
| 10.8 | `send an email to alice@example.com` | Asks: subject? then body? — NOT "what's your email?" |
| 10.9 | `reply to Alice` | Asks: which email from Alice? then what to say? |
| 10.10 | `forward an email` | Asks: which email? then to whom? |
| 10.11 | `draft an email` | Asks: to? then subject? then body? |

### Raw JSON / bad output (should NEVER appear)

| # | Type in chat | Expected |
|---|---|---|
| 10.12 | `show my unread emails` | Should NOT show `{"messages": [...], "resultSizeEstimate": 3}` raw |
| 10.13 | `read the latest email from Alice` | Should NOT show raw message_id or header JSON |

### Ambiguous input

| # | Type in chat | Expected |
|---|---|---|
| 10.14 | `email Alice` | If Alice resolves to one email → sends. If ambiguous → asks which Alice |
| 10.15 | `send email about the meeting` | Has subject hint → asks to → asks body |
| 10.16 | `check if Alice replied` | Searches for replies from Alice → shows result |

---

## SECTION 11 — NATURAL PHRASING VARIANTS

| # | Type in chat | Expected |
|---|---|---|
| 11.1 | `any unread?` | Unread email list or count |
| 11.2 | `what's in my inbox` | Inbox listing |
| 11.3 | `did Alice email me` | Searches for emails from Alice |
| 11.4 | `has anyone replied to me today` | Searches for today's replies |
| 11.5 | `drop Alice a message saying I'll be late` | Sends email to Alice with that message |
| 11.6 | `ping bob@company.com` | Sends a short email → asks for subject/body if not given |
| 11.7 | `write back to Alice` | Finds latest email from Alice → asks reply content → replies |
| 11.8 | `send a follow-up to Alice` | Searches for context → asks body → sends |
| 11.9 | `get me Alice's latest email` | Searches → reads → shows content |
| 11.10 | `let Alice know the project is done` | Sends email to Alice with that message |

---

## SECTION 12 — COMBINED JIRA + GMAIL FLOWS

| # | Type in chat | Expected flow |
|---|---|---|
| 12.1 | `send alice@example.com the details of FLAG-42` | Gets FLAG-42 from Jira → composes email with details → sends |
| 12.2 | `email Bob about the blocker issues in FLAG` | Searches Jira blockers → composes email → sends or drafts |
| 12.3 | `send a standup digest email to alice@example.com` | Gets standup digest from Jira → emails it |

---

## PASS/FAIL TRACKING

| Section | Tests | Pass | Fail | Notes |
|---|---|---|---|---|
| 1 — Read/Search | 18 | | | |
| 2 — Read Specific | 6 | | | |
| 3 — Send Email | 9 | | | |
| 4 — Reply | 4 | | | |
| 5 — Forward | 3 | | | |
| 6 — Draft | 3 | | | |
| 7 — Labels | 7 | | | |
| 8 — Threads | 3 | | | |
| 9 — Bulk Ops | 3 | | | |
| 10 — Edge Cases | 16 | | | |
| 11 — Natural Phrasing | 10 | | | |
| 12 — Combined Flows | 3 | | | |
| **TOTAL** | **85** | | | |

---

## Things to Watch For

- **Raw JSON in response** — should never show `{"messages": [...], "resultSizeEstimate": N}`. Every result must be human-readable.
- **Asking for your own email** — the agent must NEVER ask "What is your email address?" — it is auto-injected from `USER_GOOGLE_EMAIL` in `.env`.
- **Sending when you only meant to search** — "find emails about the meeting" must NOT trigger `send_gmail_message`.
- **Over-asking** — if you say "send an email to alice@example.com subject Hello body Hi", it should send in one shot with NO follow-up questions.
- **Under-asking** — if you just say "send an email", it must ask for to, subject, and body before sending anything.
- **Thread reply vs new email** — "reply to Alice" must use `thread_id` so it stays in the same Gmail thread, not start a new one.
- **Multiple match disambiguation** — if "email from Alice" returns 3 results, the agent must show a selection, not randomly pick one.
- **Slow responses** — Section 1 (read/search) should respond in under 5 seconds. Send/reply may take up to 10s.
