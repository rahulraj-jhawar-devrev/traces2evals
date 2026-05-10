import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


class LLMClient:
    """Unified LLM interface via litellm. Handles retries, JSON parsing, batch embedding."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 0.0,
        max_parallel_calls: int = 5,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_parallel_calls = max_parallel_calls

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def chat_json(self, prompt: str, model: Optional[str] = None) -> dict:
        """Call LLM and parse JSON response."""
        response = litellm.completion(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return self._parse_json(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def chat(self, prompt: str, model: Optional[str] = None) -> str:
        """Call LLM and return raw text response."""
        response = litellm.completion(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def chat_json_batch(
        self, prompts: list[str], desc: str = "LLM calls"
    ) -> list[dict]:
        """Process multiple prompts in parallel batches.

        Returns a dict for each successful prompt. On per-item failure,
        inserts None at that index so positions stay aligned.
        """
        results = [None] * len(prompts)
        for i in tqdm(range(0, len(prompts), self.max_parallel_calls), desc=desc):
            batch = prompts[i : i + self.max_parallel_calls]

            def _call_one(prompt: str) -> dict | None:
                try:
                    return self.chat_json(prompt)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "chat_json_batch item failed: %s", exc
                    )
                    return None

            with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
                batch_results = list(executor.map(_call_one, batch))
            for j, result in enumerate(batch_results):
                results[i + j] = result
        return results

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings. Supports litellm models or local sentence-transformers."""
        if self.embedding_model.startswith("local/"):
            return self._local_embed(texts)

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            response = litellm.embedding(model=self.embedding_model, input=batch)
            batch_embeddings = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)

    def _local_embed(self, texts: list[str]) -> np.ndarray:
        """Fallback to sentence-transformers for local embedding."""
        from sentence_transformers import SentenceTransformer

        model_name = self.embedding_model.split("/", 1)[1]
        model = SentenceTransformer(model_name)
        return model.encode(texts, show_progress_bar=True)

    def _parse_json(self, text: str) -> dict:
        """Strip markdown fences if present, then parse JSON."""
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return json.loads(cleaned)
