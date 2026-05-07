CLUSTER_LABELING_PROMPT = """Analyze these user queries from an AI assistant conversation system and generate a cluster label.

SAMPLE QUERIES FROM THIS CLUSTER:
{samples_formatted}

Generate:
1. A short cluster TITLE (5-8 words) that describes what users are trying to accomplish
2. A 1-2 sentence DESCRIPTION of the user intent/goal

IMPORTANT:
- Focus on USER INTENT and GOALS, not internal product features
- Use plain language that anyone would understand
- Be specific but not overly technical

Respond in JSON:
{{"title": "Your Title Here", "description": "Your description here."}}"""

PARENT_LABELING_PROMPT = """This is a BROAD CATEGORY that contains these specific sub-groups:
{subtitles_formatted}

SAMPLE QUERIES:
{samples_formatted}

Generate a high-level category name (3-5 words) that encompasses all these sub-groups.
Focus on the BROAD USER GOAL (e.g., "Automation Setup", "Data Extraction", "Debugging Issues").

Respond in JSON:
{{"title": "Broad Category Name", "description": "What users in this category are generally trying to accomplish."}}"""
