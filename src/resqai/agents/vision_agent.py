from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent
from resqai.pipelines.flood_severity.explain import explain_single_image
from resqai.pipelines.flood_severity.infer import predict_single_image
from resqai.pipelines.flood_severity.utils import resolve_from_repo

logger = logging.getLogger("resqai.agents.vision")


class VisionAgent(BaseAgent):
    name = "vision_agent"

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        image = inp.get("image")
        checkpoint = inp.get("checkpoint", Path("artifacts/flood_severity/best.pt"))
        output_dir = inp.get("output_dir", Path("outputs/explanations"))
        explain = bool(inp.get("explain", True))

        if not image:
            raise ValueError("missing_required_input:image")

        image_path = resolve_from_repo(Path(str(image)))
        ckpt_path = resolve_from_repo(Path(str(checkpoint)))
        out_dir = resolve_from_repo(Path(str(output_dir)))

        pred = predict_single_image(image_path=image_path, checkpoint_path=ckpt_path)
        if explain:
            exp = explain_single_image(
                image_path=image_path,
                checkpoint_path=ckpt_path,
                output_dir=out_dir,
            )
            pred["explainability"] = exp
        return pred

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
