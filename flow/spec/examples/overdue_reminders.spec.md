Spec: baby-spec/0.1

# System: billing-reminders
A nightly job that chases overdue invoices. The same system as the
hand-written IR2 flow in flow/examples/overdue_reminders.json.

## Constants
- FINANCE_EMAIL = "finance@example.com"
- VERY_LATE_DAYS = 30

## Entities
- run_log: run_date str, overdue int, owed int

## Services
- billing.list_overdue(as_of str) -> (invoices list of (invoice_id str, customer str, amount int, days_late int))

## Flow: overdue_reminders
Intent: Every morning, fetch overdue invoices; if there are any, email finance a summary, escalate very late ones to #collections, and log the run.
Trigger: schedule "0 6 * * *" with run_date str
Mode: async
Steps:
1. [Fetch overdue invoices] Call billing.list_overdue as fetch with as_of = run_date. Retry 2 times.
   > no "On failure": if billing is still down after 3 attempts, the whole flow fails
2. [Summarize] Compute summarize:
   - count: `count(invoices)`
   - seriously_late: `count(invoices.days_late where > VERY_LATE_DAYS)`
   - owed: the total amount owed across all invoices
   > intent; expected: `sum(invoices.amount)`
   - worst: `max(invoices.days_late)`
3. [Anything overdue?] If `count == 0`:
   a. [Nothing overdue] Stop.
4. [Email finance] Email FINANCE_EMAIL: "{count} overdue invoices, {seriously_late} over 30 days late, {owed} cents owed. Worst: {worst} days.".
5. [Over 30 days late?] If `seriously_late > 0`:
   a. [Post to #collections] Slack "#collections": "{seriously_late} invoices are more than 30 days late".
6. [Log the run] Save run_log: run_date, overdue = count, owed.
7. [Done] Stop.
