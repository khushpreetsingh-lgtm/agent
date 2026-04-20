# Jira Agent — Manual Test Plan

> **Before you start:**
> - Replace `FLAG-XX` with a real issue key from your Jira project (e.g. `FLAG-42`)
> - Replace `Sprint 23` / `Sprint 24` with real sprint names from your board
> - Replace `Tom`, `Rachel`, `Priya` with real team member names in your Jira
> - Each test is independent — start a **fresh message** for each one
> - A checkmark column is left for you to mark pass/fail

---

## SECTION 1 — DIRECT QUERIES (should be fast, no workflow)

| # | Type in chat | Expected response |
|---|---|---|
| 1.1 | `how many issues do I have` | Count of your open issues, e.g. "You have 5 open issues assigned to you." |
| 1.2 | `how many tickets assigned to me` | Same as above (different phrasing) |
| 1.3 | `count my open issues` | Same count result |
| 1.4 | `show my issues` | List of your open issues with key, summary, status, priority |
| 1.5 | `list my tickets` | Same list, different phrasing |
| 1.6 | `display my tasks` | Same list |
| 1.7 | `show all critical issues` | All issues where priority = Critical |
| 1.8 | `show all blocker issues` | All issues where priority = Blocker |
| 1.9 | `show high priority issues` | All issues where priority = High |
| 1.10 | `show medium priority tickets` | All issues where priority = Medium |
| 1.11 | `show low priority items` | All issues where priority = Low |
| 1.12 | `show all in progress issues` | Issues with status = In Progress |
| 1.13 | `what issues are in review` | Issues with status = In Review |
| 1.14 | `show issues in testing` | Issues with status = Testing |
| 1.15 | `what is in to do` | Issues with status = To Do |
| 1.16 | `show unassigned issues` | Issues with no assignee in active sprint |
| 1.17 | `show unassigned tasks` | Same |
| 1.18 | `how many blockers do we have` | Count of blocker priority issues |

**Empty state tests (expect a clean friendly sentence, NOT raw JSON):**

| # | Type in chat | Expected response |
|---|---|---|
| 1.19 | `show all done issues` | "No issues marked as Done yet." OR a list if some exist |
| 1.20 | `show all blocker issues` | If none: "No blockers right now — all clear!" |
| 1.21 | `show in review issues` | If none: "Nothing is currently in review." |
| 1.22 | `show unassigned issues` | If none: "No unassigned issues found in the sprint." |

---

## SECTION 2 — ASSIGNEE WORKLOAD QUERIES

| # | Type in chat | Expected response |
|---|---|---|
| 2.1 | `what is [real team member name] working on` | Their open issues listed |
| 2.2 | `show me [name]'s issues` | Their issue list |
| 2.3 | `what does [name] have assigned` | Their issue list |
| 2.4 | `who has the most work right now` | Should attempt to show workload distribution |
| 2.5 | `show team workload` | Team workload summary |

---

## SECTION 3 — STATUS TRANSITIONS

> For each test below, use a real issue key from your project.

| # | Type in chat | Expected flow |
|---|---|---|
| 3.1 | `Move FLAG-XX to Done` | Fetches transitions → executes → confirms Done |
| 3.2 | `Mark FLAG-XX as complete` | Same |
| 3.3 | `Close out FLAG-XX` | Same |
| 3.4 | `FLAG-XX is finished` | Same |
| 3.5 | `Start work on FLAG-XX` | Moves to In Progress |
| 3.6 | `Move FLAG-XX to In Progress` | Same |
| 3.7 | `Mark FLAG-XX as started` | Same |
| 3.8 | `Move FLAG-XX to Review` | Moves to In Review |
| 3.9 | `Send FLAG-XX for PR review` | Same |
| 3.10 | `Move FLAG-XX to Testing` | Moves to Testing |
| 3.11 | `Send FLAG-XX to QA` | Same |
| 3.12 | `Move FLAG-XX to Backlog` | Moves to Backlog |
| 3.13 | `Close FLAG-XX` | Closes/Resolves |
| 3.14 | `Mark FLAG-XX as resolved` | Same |
| 3.15 | `Reopen FLAG-XX` | Reopens issue |
| 3.16 | `Update ticket status` | Should ask which ticket → then ask which status (selection) |
| 3.17 | `Transition an issue` | Should ask which ticket → then which status |
| 3.18 | `Change the status of FLAG-XX` | Should show status options as buttons to pick from |

---

## SECTION 4 — PRIORITY CHANGES

| # | Type in chat | Expected flow |
|---|---|---|
| 4.1 | `Set FLAG-XX priority to Critical` | Updates priority to Critical, confirms |
| 4.2 | `Mark FLAG-XX as critical` | Same |
| 4.3 | `Set FLAG-XX to high priority` | Updates to High |
| 4.4 | `Set FLAG-XX priority to Blocker` | Updates to Blocker |
| 4.5 | `Mark FLAG-XX as blocker` | Same |
| 4.6 | `Set FLAG-XX to medium priority` | Updates to Medium |
| 4.7 | `Set FLAG-XX to low priority` | Updates to Low |
| 4.8 | `Escalate FLAG-XX` | Should update to Critical or Blocker and confirm |
| 4.9 | `Set priority to high` | Should ask which issue first |

---

## SECTION 5 — ASSIGN ISSUE

| # | Type in chat | Expected flow |
|---|---|---|
| 5.1 | `Assign FLAG-XX to me` | Assigns to your configured Jira username, confirms |
| 5.2 | `Take FLAG-XX` | Same |
| 5.3 | `Put FLAG-XX on my plate` | Same |
| 5.4 | `Assign FLAG-XX to [real name in your Jira]` | Searches user → assigns → confirms with display name |
| 5.5 | `Reassign FLAG-XX to [name]` | Same as assign |
| 5.6 | `Change assignee of FLAG-XX to [name]` | Same |
| 5.7 | `Assign FLAG-XX` | Should ask who to assign to |

---

## SECTION 6 — LOG TIME

| # | Type in chat | Expected flow |
|---|---|---|
| 6.1 | `Log 3 hours on FLAG-XX` | Logs 3h, confirms |
| 6.2 | `Log 2.5 hours on FLAG-XX` | Logs 2h 30m, confirms |
| 6.3 | `Log 30 minutes on FLAG-XX` | Logs 30m, confirms |
| 6.4 | `Log 1 hour on FLAG-XX` | Logs 1h, confirms |
| 6.5 | `Log 4 hours on FLAG-XX for API implementation` | Logs 4h with comment "API implementation" |
| 6.6 | `Log 2 hours on FLAG-XX investigating root cause` | Logs 2h with comment |
| 6.7 | `Log time on FLAG-XX` | Should ask how many hours |
| 6.8 | `Log work on FLAG-XX` | Should ask how many hours |

---

## SECTION 7 — ADD COMMENT

| # | Type in chat | Expected flow |
|---|---|---|
| 7.1 | `Add a comment to FLAG-XX` | Asks what to comment → adds → confirms |
| 7.2 | `Comment on FLAG-XX` | Same |
| 7.3 | `Add a comment to FLAG-XX: Ready for QA review` | Adds comment directly without asking, confirms |
| 7.4 | `Add comment to FLAG-XX: Needs design approval` | Adds comment directly |
| 7.5 | `Add a note to FLAG-XX: Waiting on customer feedback` | Adds comment |

---

## SECTION 8 — LINK ISSUES

> Use two real issue keys for these tests.

| # | Type in chat | Expected flow |
|---|---|---|
| 8.1 | `FLAG-XX is blocked by FLAG-YY` | Creates "is blocked by" link, confirms |
| 8.2 | `FLAG-XX blocks FLAG-YY` | Creates "blocks" link, confirms |
| 8.3 | `FLAG-XX is duplicate of FLAG-YY` | Creates duplicate link, confirms |
| 8.4 | `FLAG-XX relates to FLAG-YY` | Creates "relates to" link, confirms |
| 8.5 | `Link FLAG-XX to FLAG-YY` | Should ask which type of link (selection buttons) |

---

## SECTION 9 — MOVE TO SPRINT

| # | Type in chat | Expected flow |
|---|---|---|
| 9.1 | `Move FLAG-XX to the current sprint` | Fetches boards → sprints → selection → moves → confirms |
| 9.2 | `Add FLAG-XX to sprint` | Should show sprint selection buttons |
| 9.3 | `Move FLAG-XX to backlog` | Moves issue out of sprint to backlog |
| 9.4 | `Move FLAG-XX to sprint` | Should ask which sprint (selection) |

---

## SECTION 10 — CREATE TICKET

| # | Type in chat | Expected flow |
|---|---|---|
| 10.1 | `Create a bug for the login page crash on mobile` | Selects project → creates Bug with that summary |
| 10.2 | `Log a bug: Mobile Safari session timeout` | Creates Bug |
| 10.3 | `Create a story for the new dashboard redesign` | Creates Story |
| 10.4 | `Create a task to update documentation` | Creates Task |
| 10.5 | `Create a new ticket` | Asks project → asks issue type → asks summary → creates |
| 10.6 | `Create a Jira issue` | Same |
| 10.7 | `Log an issue` | Same |
| 10.8 | `Create story: Dark mode support` | Creates Story with that summary, asks project |

---

## SECTION 11 — CREATE SUBTASK

| # | Type in chat | Expected flow |
|---|---|---|
| 11.1 | `Create a subtask for FLAG-XX` | Asks subtask title → creates Sub-task under FLAG-XX |
| 11.2 | `Add a subtask to FLAG-XX for writing unit tests` | Creates Sub-task with summary "writing unit tests" |
| 11.3 | `Create subtask under FLAG-XX: API documentation` | Creates directly |
| 11.4 | `Create a subtask` | Should ask which parent issue first |

**Label-bleed check for 11.1:**
- When a selection dropdown appears, pick the parent issue
- Verify the subtask title input is NOT pre-filled with the option label text

---

## SECTION 12 — CREATE SPRINT

| # | Type in chat | Expected flow |
|---|---|---|
| 12.1 | `Create a sprint` | Asks project → asks board → asks sprint name → creates → shows sprint details |
| 12.2 | `Create a new sprint in FLAG` | Should skip project selection if FLAG is clear |

---

## SECTION 13 — SPRINT START / CLOSE

| # | Type in chat | Expected flow |
|---|---|---|
| 13.1 | `Start Sprint [name]` | Fetches boards → sprints → selection → starts → confirms |
| 13.2 | `Begin Sprint [name]` | Same |
| 13.3 | `Close Sprint [name]` | Fetches → selection → closes → confirms |
| 13.4 | `End Sprint [name]` | Same |
| 13.5 | `Complete Sprint [name]` | Same |

---

## SECTION 14 — STANDUP & BRIEFING

| # | Type in chat | Expected response |
|---|---|---|
| 14.1 | `Generate standup digest` | Shows: In Progress issues, Done yesterday, Active Blockers — all in one message |
| 14.2 | `Show standup digest` | Same |
| 14.3 | `Standup` | Same |
| 14.4 | `Brief me for today` | Similar digest view |
| 14.5 | `Daily briefing` | Same |

---

## SECTION 15 — EDGE CASES (Intent Boundary Tests)

### Should NOT create a ticket

| # | Type in chat | Expected |
|---|---|---|
| 15.1 | `How many tasks do I have open?` | Returns count — does NOT trigger create_ticket |
| 15.2 | `What tasks are unassigned?` | Returns unassigned list — does NOT trigger create_ticket |

### Should create a ticket

| # | Type in chat | Expected |
|---|---|---|
| 15.3 | `Create a new task for user authentication` | Creates Task |
| 15.4 | `File a bug for login crash` | Creates Bug |

### Missing field — should ask

| # | Type in chat | Expected |
|---|---|---|
| 15.5 | `Set priority to high` | Should ask: which issue? |
| 15.6 | `Move to done` | Should ask: which issue? |
| 15.7 | `Assign FLAG-XX` | Should ask: who to assign to? |
| 15.8 | `Log time on FLAG-XX` | Should ask: how many hours? |
| 15.9 | `Add a comment to FLAG-XX` | Should ask: what to comment? |
| 15.10 | `Move FLAG-XX to sprint` | Should show sprint selection |
| 15.11 | `Create a subtask` | Should ask: which parent issue? |
| 15.12 | `Link FLAG-XX to FLAG-YY` | Should ask: what type of link? |

### Ambiguous / non-Jira IDs (should NOT treat as issue keys)

| # | Type in chat | Expected |
|---|---|---|
| 15.13 | `What is wrong with PROJECT-001?` | Should answer conversationally OR look up as Jira key — NOT crash |
| 15.14 | `Fix PROJ-XXX` | Should handle gracefully (XXX is not a valid ID) |

### Multi-issue operations

| # | Type in chat | Expected |
|---|---|---|
| 15.15 | `Move FLAG-XX and FLAG-YY to done` | Should handle both — transition each one |

---

## SECTION 16 — ROBUSTNESS / PHRASING VARIANTS

These test that the agent handles natural phrasing without strict keywords.

| # | Type in chat | Expected |
|---|---|---|
| 16.1 | `which all open tasks do i have` | Your open task list |
| 16.2 | `can you show me what i need to work on` | Your open issues |
| 16.3 | `what's on my plate` | Your open issues |
| 16.4 | `give me a quick summary of my work` | Your open issues (or standup-style) |
| 16.5 | `are there any blockers in the sprint` | Blocker priority issues |
| 16.6 | `i need to mark FLAG-XX as finished` | Transitions to Done |
| 16.7 | `please add a note to FLAG-XX saying it needs review` | Adds comment |
| 16.8 | `i worked 2 hours on FLAG-XX` | Logs 2h |
| 16.9 | `FLAG-XX is dependent on FLAG-YY` | Should create a link (relates to or blocked by) |
| 16.10 | `reassign FLAG-XX` | Asks who to reassign to |

---

## PASS/FAIL TRACKING

| Section | Tests | Pass | Fail | Notes |
|---|---|---|---|---|
| 1 — Direct Queries | 22 | | | |
| 2 — Assignee Workload | 5 | | | |
| 3 — Status Transitions | 18 | | | |
| 4 — Priority Changes | 9 | | | |
| 5 — Assign Issue | 7 | | | |
| 6 — Log Time | 8 | | | |
| 7 — Add Comment | 5 | | | |
| 8 — Link Issues | 5 | | | |
| 9 — Move to Sprint | 4 | | | |
| 10 — Create Ticket | 8 | | | |
| 11 — Create Subtask | 4 | | | |
| 12 — Create Sprint | 2 | | | |
| 13 — Sprint Start/Close | 5 | | | |
| 14 — Standup/Briefing | 5 | | | |
| 15 — Edge Cases | 15 | | | |
| 16 — Robustness | 10 | | | |
| **TOTAL** | **137** | | | |

---

## Things to Watch For

- **Raw JSON in the response** — should never appear. If you see `{"total": -1, ...}` that's a bug.
- **"No issues found."** with a prefix — should be one clean sentence like "You don't have any open tasks right now."
- **Hallucinated issue keys** — agent should never make up an issue key like FLAG-999 if not asked to
- **Asking for things already given** — if you say "Assign FLAG-42 to me", it should NOT ask for the issue key again
- **Slow responses** — Section 1 (direct queries) should respond in under 5 seconds. Others are slower due to planning.
- **Wrong tool called** — watch the step logs. If a step says `[browser_act]` for a Jira task, that's wrong.
