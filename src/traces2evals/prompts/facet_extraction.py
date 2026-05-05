FACET_EXTRACTION_PROMPT = """What is the topic of this conversation? Answer with a privatized summary that omits any private information and captures both sides — what the user asked and what the assistant did in response.

The conversation has {num_turns} total turns. If the user discusses multiple clearly unrelated topics, produce one facet per topic.

Conversation:
{conversation_text}

Respond in JSON:
{{
  "facets": [
    {{
      "summary": "2-3 sentence privatized summary",
      "num_turns": <turns spent on this facet>
    }}
  ]
}}

Privatization rules — the summary MUST NOT contain:
- Names of people, companies, or organizations
- Email addresses, phone numbers, account/ticket/issue IDs
- URLs, IP addresses, API keys, credentials
- Specific project, repository, or product names
Replace with generic descriptions (e.g., "a user" not a name, "an internal tool" not a product name).

Keep summaries crisp — focus on what the user is trying to accomplish. Also briefly capture what the assistant did (retrieved data, generated content, ran a query, made an update, explained something, or declined) and whether the task was completed.

Most conversations should produce a SINGLE facet. Only split when topics are clearly unrelated."""
