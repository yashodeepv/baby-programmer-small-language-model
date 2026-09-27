Spec: baby-spec/0.1

# System: leave-desk
Employees request leave. Short leave within balance is approved at once;
long leave for junior staff waits for HR review. This example exercises
constants, a named call, both kinds of validation failure, an If with an
Otherwise, and intent logic.

## Constants
- LONG_LEAVE_DAYS = 10
- SENIOR_LEVEL = 5
- HR_EMAIL = "hr@corp.example"

## Entities
- leave_request: employee_id str, days int, status str

## Services
- hris.get_employee(employee_id str) -> (department str, level int, active bool)

## Flow: request_leave
Intent: Approve or queue a leave request and tell the manager.
Trigger: POST /leave-requests with employee_id str, employee_email str, manager_email str, days int, balance int, kind str
Mode: sync
Steps:
1. [Valid number of days?] Validate: `days > 0`. Otherwise stop with error "days must be positive".
2. [Look up the employee] Call hris.get_employee as employee with employee_id. Retry 2 times. On failure stop with error "employee lookup failed", responding with employee_id.
3. [Still employed?] Validate: `employee.active`. Otherwise fail with "inactive employee".
   > "fail with": no end step; the whole flow fails and the caller sees a failure
4. [Decide] Compute decision:
   - enough_balance: days is at most balance, or kind is "sick"
   > intent; expected: `days <= balance or kind == "sick"`
   - needs_review: `days >= LONG_LEAVE_DAYS and employee.level < SENIOR_LEVEL`
5. [Enough balance?] If `not enough_balance`:
   a. [Not enough balance] Stop with error "not enough leave balance", responding with employee_id, balance.
6. [Needs HR review?] If `needs_review`:
   a. [Queue for review] Save leave_request: employee_id, days, status = "pending".
   b. [Ask HR] Email HR_EMAIL: "Please review {days} days of {kind} leave for {employee_id}".
   Otherwise:
   a. [Approve] Save leave_request: employee_id, days, status = "approved".
7. [Tell the manager] Email manager_email: "{employee_id} requested {days} days of {kind} leave".
8. [Request recorded] Respond with employee_id, days, needs_review.
