"""app - Spec Studio: paste a spec, get a flowchart, test runs and verified code.

    python3.11 -m flow.app

    pipeline.py   spec -> compile -> test runs -> code -> verification
    prompt.py     the prompt a user gives an LLM to get a spec back
    server.py     local HTTP server (standard library only)
    static/       the single-page UI
"""
