EVAL_CRAFTING_PROMPT = """You are generating an eval test case from a production AI assistant session.

CLUSTER: "{cluster_title}" — {cluster_description}

SESSION (conversation):
{conversation_text}

Your task:
1. Determine if this is SINGLE-TURN or MULTI-TURN.
2. For multi-turn: synthesize the user's complete intent into ONE self-contained query.
   For single-turn: extract the user message verbatim.
3. Write the reference response (what a correct response looks like).
4. Write judge remarks (what capability is being tested).

Respond in JSON:
{{
  "user_message": "the query (privatized, self-contained)",
  "reference_response": "what a correct response looks like",
  "synthesis_type": "verbatim" or "synthesized",
  "judge_remarks": "1-3 sentences — what competent handling looks like for this specific question",
  "capability_tested": "short description of the agent capability being evaluated"
}}

Rules for user_message:
- Self-contained (no pronouns referencing prior context)
- Privatized (no real names, emails, IDs — use generic descriptions)
- If synthesized from multi-turn, reads naturally as something a user would actually type

Rules for reference_response:
- Demonstrates what a correct, complete answer looks like
- Privatized (no real PII)
- Captures the key information the agent should provide

Rules for judge_remarks:
- Frame as behavioral capability ("what competence looks like")
- Never use boilerplate ("directly address the user's request", "accurate information")
- Specific enough that a judge can distinguish good from mediocre
- Never prescribe exact output content — describe the SHAPE of competence, not the content"""
