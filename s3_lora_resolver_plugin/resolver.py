import os
import s3fs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

class S3LoRAResolver(LoRAResolver):
    def __init__(self):
        self.s3 = s3fs.S3FileSystem()
        self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
        self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

    async def resolve_lora(self, base_model_name, lora_name):
        s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
        local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

        # Download the LoRA from S3 to the local path
        await self.s3._get(
            s3_path, local_path, recursive=True, maxdepth=1
        )

        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_path=local_path,
            lora_int_id=abs(hash(lora_name))
        )
        return lora_request
    
def register():
    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)