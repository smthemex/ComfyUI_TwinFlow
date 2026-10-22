
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
from .twinflow_core import TwinFlow_ZImage_Loader, TwinFlow_QwenImage_Loader, TwinFlow_Sampler_Select


class TwinFlow_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TwinFlow_ZImage_Loader,
            TwinFlow_QwenImage_Loader,
            TwinFlow_Sampler_Select,
        ]


async def comfy_entrypoint() -> TwinFlow_SM_Extension:
    return TwinFlow_SM_Extension()
