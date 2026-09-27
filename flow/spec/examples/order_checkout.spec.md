Spec: baby-spec/0.1

# System: shop-orders
Takes orders over HTTP, charges them and confirms them. The same system as
the hand-written IR2 flow in flow/examples/order_checkout.json.

## Constants
- LARGE_ORDER = 100000
- SALES_CHANNEL = "#sales"

## Entities
- order: order_id str, total int, charge_id str

## Services
- payments.charge(order_id str, amount int) -> (charge_id str, status str)

## Flow: order_checkout
Intent: Reject empty orders, price the order with a gold discount, charge it, alert sales about large orders, then confirm and save it.
Trigger: POST /orders with order_id str, customer_email str, tier str, items list of (sku str, amount int)
Mode: sync
Steps:
1. [Has items?] Validate: `len(items) > 0`. Otherwise stop with error "order has no items".
2. [Price the order] Compute pricing:
   - subtotal: the sum of the item amounts
   > intent, programmed by baby-programmer; expected: `sum(items.amount)`
   - discount: `subtotal / 10 if tier == "gold" else 0`
   - total: `subtotal - discount`
3. [Charge the card] Call payments.charge with order_id, amount = total. Retry 1 time. On failure stop with error "payment failed", responding with order_id.
4. [$1000 or more?] If `total >= LARGE_ORDER`:
   a. [Tell sales] Slack SALES_CHANNEL: "Large order {order_id}: {total} cents".
5. [Confirm and save] In parallel:
   a. [Email the customer] Email customer_email: "Order {order_id} confirmed. Total: {total}".
   b. [Save the order] Save order: order_id, total, charge_id.
6. [Order placed] Respond with order_id, charge_id, total.
