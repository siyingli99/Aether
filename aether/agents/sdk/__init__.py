"""Aether Adapter SDK.

Authors of new AD adapters can subclass :class:`BaseAgent` and call
:func:`serve` to wire it up to the FIFO protocol used by ``aether/closed_loop``::

    from aether.agents.sdk import BaseAgent, serve

    class MyAgent(BaseAgent):
        def setup(self):
            self.model = load_my_model()

        def infer(self, obs):
            plan = self.model(obs["rgb_front"])
            return {"plan_traj": plan}  # (T, 2)

    if __name__ == "__main__":
        serve(MyAgent())

The shell entrypoint then becomes a one-liner::

    python -u my_agent.py "$1" "$2"   # ${cuda_id} ${output_dir}
"""
from .bridge import BaseAgent, serve

__all__ = ["BaseAgent", "serve"]
