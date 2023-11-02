import os

import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.utils.generic_utils import try_to_import_replicate

_VALID_STABLE_DIFFUSION_MODEL = [
    "sdxl:af1a68a271597604546c09c64aabcd7782c114a63539a4a8d14d1eeda5630c33",
    "lora-training:a5b0e981c875f656936c6c67b385c27057e226141e4e62fd5177ce96caee95e2"
]

class StableDiffusionLoRA(AbstractFunction):
    def setup(
        self,
        lora_model="lora-training:a5b0e981c875f656936c6c67b385c27057e226141e4e62fd5177ce96caee95e2",
        task_type="style", 
        file_location='/Users/kacylombard/Desktop/evadb/evadb/tutorials/lorazip.zip',
    ) -> None:
        assert (
            lora_model in _VALID_STABLE_DIFFUSION_MODEL
        ), f"Unsupported Stable Diffusion {lora_model}"
        self.lora_model = lora_model
        self.task_type = task_type
        self.file_location = file_location

    @property
    def name(self) -> str:
        return "StableDiffusionLoRA"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["name", "data"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.UINT8,
                ],
                column_shapes=[(19,2)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["url"],
                column_types=[
                    NdArrayType.STR,
                ],
                column_shapes=[(1,1)],
            )
        ],
    )
    def forward(self, df) -> pd.DataFrame():
        # task_type (string): face, object, style; Type of LoRA model you want to train
        # seed (int): a seed for reproducible training
        # file_location (string): location of instance_data file (ZIP file with training images)
        # resolution (int): resolution of input images (default 512); images will be resized to this value
        import pathlib
        from PIL import Image
        from urllib.request import urlretrieve
        from diffusers import StableDiffusionPipeline
        import torch
        os.mkdir('/Users/kacylombard/Desktop/evadb/evadb/tutorials/lora_images/') 
        print(df.info)
        for index, row in df.iterrows():
            print(row)
            im = Image.fromarray(row["data"])
            nameList = row["name"].split('/')
            im.save('/Users/kacylombard/Desktop/evadb/evadb/tutorials/lora_images/' + nameList[-1])
        
        import shutil
        shutil.make_archive("lorazip", 'zip', '/Users/kacylombard/Desktop/evadb/evadb/tutorials/lora_images/')
        try_to_import_replicate()
        import replicate

        if os.environ.get("REPLICATE_API_TOKEN") is None:
            replicate_api_key = (
                "r8_Q75IAgbaHFvYVfLSMGmjQPcW5uZZoXz0jGalu"  # token for testing
            )
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
        def train_model_on_images(): 
            output = replicate.run(
                "cloneofsimo/" + self.lora_model,
                input={"instance_data": open(self.file_location, "rb"), "task": self.task_type}
            )
            return output
        dfOut = train_model_on_images()
        output = replicate.run(
            "cloneofsimo/lora:fce477182f407ffd66b94b08e761424cabd13b82b518754b83080bc75ad32466",
            input={"prompt": "A painting of the ancient Greece, ultra-realism"},
            lora_urls=dfOut
        )
        print(output)
        return pd.DataFrame({"url": output}) 
