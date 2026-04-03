import pytest
import json

from paloalto.agent.judge import AgentJudge, AgentAction


class TestAgentAction:
    def test_accept_action(self):
        action = AgentAction(action="accept", reasoning="Looks good")
        assert action.action == "accept"
        assert action.params is None

    def test_modify_action(self):
        action = AgentAction(
            action="modify",
            params={"n_neighbors": 20},
            reasoning="Higher n_neighbors based on trend",
        )
        assert action.params["n_neighbors"] == 20

    def test_prune_action(self):
        action = AgentAction(
            action="prune",
            new_bounds={"min_dist": [0.01, 0.5]},
            reasoning="Good results concentrated in low min_dist",
        )
        assert action.new_bounds["min_dist"] == [0.01, 0.5]


class TestAgentJudge:
    def test_build_context(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        context = judge.build_context(
            trial_number=6,
            trial_history=[
                {"params": {"n_neighbors": 10}, "scores": {"scib_overall": 0.5, "scgraph_score": 0.6}},
            ],
            current_best={"params": {"n_neighbors": 10}, "scores": {"scib_overall": 0.5, "scgraph_score": 0.6}},
            bo_suggestion={"n_neighbors": 15, "min_dist": 0.2},
            search_space={"n_neighbors": {"type": "int", "bounds": [5, 200]}},
            dataset_summary={"n_cells": 10000, "n_batches": 3, "n_types": 8},
        )
        assert "trial_number" in context
        assert context["trial_number"] == 6

    def test_parse_response_accept(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        raw = '{"action": "accept", "reasoning": "Suggestion looks reasonable"}'
        action = judge.parse_response(raw)
        assert action.action == "accept"

    def test_parse_response_fallback(self):
        judge = AgentJudge(model="claude-sonnet-4-6")
        action = judge.parse_response("this is not valid json at all")
        assert action.action == "accept"  # safe fallback
