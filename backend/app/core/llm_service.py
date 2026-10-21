import json
import httpx
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from app.config import settings

class OllamaGemmaService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model_name: str = settings.OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def serialize_shap_context(
        self,
        domain_context: str,
        prediction: float,
        base_value: float,
        feature_contributions: List[Dict[str, Any]],
        top_k: int = 5
    ) -> str:
        """
        Extracts top-k positive drivers and negative mitigators, serializing
        them into an immutable JSON payload to prevent model hallucinations.
        """
        positive_factors = [f for f in feature_contributions if f["shap_value"] > 0]
        negative_factors = [f for f in feature_contributions if f["shap_value"] < 0]

        # Sort by absolute SHAP attribution
        positive_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        negative_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        payload = {
            "domain_of_expertise": domain_context,
            "reference_baseline_expectation": round(base_value, 4),
            "model_prediction_output": round(prediction, 4),
            "top_positive_contributions": positive_factors[:top_k],
            "top_negative_contributions": negative_factors[:top_k]
        }
        return json.dumps(payload, indent=2)

    async def generate_explanation(
        self,
        domain_context: str,
        serialized_context: str,
        user_query: Optional[str] = None
    ) -> str:
        """
        Calls local Ollama daemon hosting Gemma with strict system prompt boundaries.
        """
        system_prompt = (
            f"You are an expert Explainable AI (XAI) consultant specializing in {domain_context}.\n"
            "Your objective is to translate technical SHAP (SHapley Additive exPlanations) values "
            "into clear, professional, plain-language insights for non-technical stakeholders.\n"
            "STRICT RULES:\n"
            "1. Ground every statement strictly in the provided JSON attribution context.\n"
            "2. Never fabricate feature names or numeric values outside the payload.\n"
            "3. Format your response into 4 distinct sections:\n"
            "   - Executive Summary: Compare the model prediction against the baseline.\n"
            "   - Primary Upward Pressures: Explain the leading factors increasing the score.\n"
            "   - Mitigating Downward Pressures: Explain the factors decreasing or stabilizing the score.\n"
            "   - Actionable Domain Next Step: Concrete operational recommendation."
        )

        if user_query:
            prompt_content = (
                f"Context Data:\n{serialized_context}\n\n"
                f"Follow-up Stakeholder Question: {user_query}\n\n"
                "Provide a direct, grounded answer relying only on the verified feature contributions."
            )
        else:
            prompt_content = (
                f"Context Data:\n{serialized_context}\n\n"
                "Generate the full 4-part structured explanation for this prediction."
            )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_content}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 512
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to Ollama daemon at {self.base_url}. Ensure 'ollama serve' is running."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ollama inference error: {str(e)}"
            )

llm_service = OllamaGemmaService()