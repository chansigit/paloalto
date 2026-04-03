"""LLM Agent Judge for BO trial review."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from paloalto.utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an expert judge for Bayesian optimization of single-cell embedding visualizations.

You review each BO trial: the suggested hyperparameters, the trial history, and the current best.
You can take one of these actions:

- ACCEPT: Use the BO suggestion as-is. Choose this when the suggestion looks reasonable.
- MODIFY: Adjust specific parameters. Provide the modified params dict. Stay within search space bounds.
- INJECT: Replace the suggestion entirely with your own candidate. Stay within bounds.
- PRUNE: Shrink the search space bounds for one or more parameters. Only shrink, never expand.
- STOP: Recommend early stopping if the optimization has converged.

Respond with a JSON object:
{
    "action": "accept" | "modify" | "inject" | "prune" | "stop",
    "params": {...},           // for modify/inject only
    "new_bounds": {...},       // for prune only, e.g. {"min_dist": [0.01, 0.5]}
    "reasoning": "..."         // required: explain your decision
}

Guidelines:
- If recent trials show a clear trend (e.g., lower min_dist is always better), consider PRUNE or MODIFY.
- If the BO is exploring a region that has consistently produced poor results, MODIFY away from it.
- If the best score hasn't improved in 5+ trials and the Pareto front is stable, consider STOP.
- Be conservative: ACCEPT is the default. Only intervene when you have clear evidence.
- Look for patterns in what makes good vs. bad embeddings for this dataset.
"""


@dataclass
class AgentAction:
    action: str  # accept, modify, inject, prune, stop
    params: Optional[Dict] = None
    new_bounds: Optional[Dict] = None
    reasoning: str = ""


class AgentJudge:
    """LLM-based judge for reviewing BO trial suggestions."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None):
        self.model = model
        self.api_key = api_key
        self.log: List[Dict] = []

    def build_context(
        self,
        trial_number: int,
        trial_history: List[Dict],
        current_best: Dict,
        bo_suggestion: Dict,
        search_space: Dict,
        dataset_summary: Dict,
    ) -> Dict:
        """Build the context dict for the agent prompt."""
        # Compute recent trend
        if len(trial_history) >= 3:
            recent_scores = [
                t["scores"].get("scib_overall", 0) + t["scores"].get("scgraph_score", 0)
                for t in trial_history[-3:]
            ]
            if recent_scores[-1] > recent_scores[0]:
                trend = "improving"
            elif recent_scores[-1] < recent_scores[0] - 0.05:
                trend = "degrading"
            else:
                trend = "flat"
        else:
            trend = "too_early"

        return {
            "trial_number": trial_number,
            "trial_history": trial_history,
            "current_best": current_best,
            "bo_suggestion": bo_suggestion,
            "search_space": search_space,
            "dataset_summary": dataset_summary,
            "recent_trend": trend,
        }

    def review(self, context: Dict) -> AgentAction:
        """Call the LLM to review the trial. Returns an AgentAction."""
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed, defaulting to ACCEPT")
            return AgentAction(action="accept", reasoning="anthropic not available")

        client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()

        user_msg = json.dumps(context, indent=2, default=str)

        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw_text = response.content[0].text
        action = self.parse_response(raw_text)

        self.log.append({
            "trial_number": context["trial_number"],
            "action": action.action,
            "reasoning": action.reasoning,
            "params": action.params,
            "new_bounds": action.new_bounds,
        })

        logger.info(f"  Agent [{action.action}]: {action.reasoning[:100]}")
        return action

    def parse_response(self, raw: str) -> AgentAction:
        """Parse JSON response from LLM, with fallback to ACCEPT."""
        try:
            # Handle markdown code blocks
            if "```" in raw:
                start = raw.index("```") + 3
                if raw[start:start + 4] == "json":
                    start += 4
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            data = json.loads(raw)
            return AgentAction(
                action=data.get("action", "accept"),
                params=data.get("params"),
                new_bounds=data.get("new_bounds"),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse agent response, defaulting to ACCEPT")
            return AgentAction(action="accept", reasoning="Parse failure, defaulting to accept")

    def get_log(self) -> List[Dict]:
        return self.log
