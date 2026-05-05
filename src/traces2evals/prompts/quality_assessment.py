QUALITY_ASSESSMENT_PROMPT = """Read this conversation between a user and an AI assistant. Assess the quality of the interaction.

Conversation:
{conversation_text}

Respond in JSON:
{{
  "task_completed": true or false,
  "response_quality": 1-5,
  "is_noise": true or false,
  "brief_assessment": "one sentence"
}}

Definitions:
- task_completed: Did the assistant accomplish what the user asked?
- response_quality: 1=terrible/wrong, 2=poor, 3=acceptable, 4=good, 5=excellent
- is_noise: Is this a greeting, test message, gibberish, or non-substantive interaction with no real task?"""
