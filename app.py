import logging
import os
from typing import Dict, List, Optional

import lightning as L
from lightning.app.frontend import StaticWebFrontend
from rich import print
from rich.logging import RichHandler

from diffusion_app.components.model_demo import ModelDemo

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)


class DiffusionApp(L.LightningFlow):
    """Demo of stable diffusion - img2img
    training_log_url: [Optional] Link for experiment manager like wandb or tensorboard
    launch_gradio: Launch Gradio demo. Defaults to False. You should update the
        `diffusion_app/components/model_demo.py` file to your use case.
    tab_order: You can optionally reorder the tab layout by providing a list of tab name.
    """

    def __init__(
        self,
        training_log_url: Optional[str] = None,
        launch_gradio: bool = False,
        tab_order: Optional[List[str]] = None,
    ) -> None:

        super().__init__()
        self.training_logs = training_log_url
        self.model_demo = None
        self.tab_order = tab_order

        if launch_gradio:
            self.model_demo = ModelDemo()

    def run(self) -> None:
        if os.environ.get("TESTING_LAI"):
            print("⚡ Lightning Diffusion App! ⚡")
        if self.model_demo:
            self.model_demo.run()

    def configure_layout(self) -> List[Dict[str, str]]:
        tabs = []

        if self.training_logs:
            tabs.append({"name": "Training Logs", "content": self.training_logs})

        if self.model_demo:
            tabs.append({"name": "Diffusion Demo", "content": self.model_demo.url})

        return self._order_tabs(tabs)

    def _order_tabs(self, tabs: List[dict]):
        """Reorder the tab layout."""
        if self.tab_order is None:
            return tabs
        order_int: Dict[str, int] = {e.lower(): i for i, e in enumerate(self.tab_order)}
        try:
            return sorted(tabs, key=lambda x: order_int[x["name"].lower()])
        except KeyError as e:
            logger.error(
                f"One of the key '{e.args[0]}' that you passed as `tab_order` argument is missing or incorrect. "
                f"Please check {tabs}"
            )


if __name__ == "__main__":
    tabs = ["Diffusion Demo"]

    app = L.LightningApp(
        DiffusionApp(
            launch_gradio=True,
            tab_order=tabs,
        )
    )
