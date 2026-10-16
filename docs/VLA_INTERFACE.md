# Model-independent VLA extension

The original Apache-licensed RPent interfaces remain at:

- `third_party/rpent/rpent/robots/components/vla_client_base.py`: `BaseVLAClient.predict(obs, options=None)` calls `vla.predict` with a 120-second inference timeout.
- `third_party/rpent/rpent/robots/components/vla_facade_base.py`: subclass `BaseVLAFacade` and implement `predict`; it registers the `vla.predict` RPC route and retains the underlying optional session lifecycle.
- `third_party/rpent/rpent/robots/components/action_spec.py`: generic action description helpers.
- `third_party/rpent/rpent/utils/rpc/`: existing socket and HTTP transports.

From a source checkout:

```python
import settings  # Adds the bundled RPent core to sys.path.
from rpent.robots.components import BaseVLAClient, BaseVLAFacade

class MyPolicy(BaseVLAFacade):
    def predict(self, obs, options=None):
        return self.model.predict(obs, options)  # Load your own model in __init__.
```

This is an interface example, not an implemented model or server startup recipe. The observation schema and returned action format belong to the selected policy. Before routing output to HRC, define and validate frame, units, quaternion order, control period, gripper convention, state/image preprocessing and action bounds. The existing HRC tools use metres, world coordinates, wxyz quaternions and physical gripper widths; normalized model output cannot be forwarded without an explicit adapter.

No provider is silently selected, no checkpoint is downloaded, and no learned grasp tool is exposed by the default Toolkit. Adding a provider must preserve current public/private observation and native emergency-stop boundaries.
