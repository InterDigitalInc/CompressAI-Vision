from typing import Dict

from compressai_vision.codec import Bypass
from compressai_vision.model_wrappers import BaseWrapper
from compressai_vision.registry import register_codec


@register_codec("cfp_codec")
class CFP_CODEC(Bypass):
    """Encoder / Decoder class for FCVCM CfP"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def encode(
        self,
        input: Dict,
        file_prefix: str = "",
    ) -> Dict:
        """
        Bypass encoder
        Returns the input and calculates its raw size
        """
        del file_prefix  # used in other codecs that write bitstream files

        print("Encode")
        for layer_name, layer_data in input["data"].items():
            print(layer_name)
            print(layer_data.shape)

        return {
            "bytes": [
                1,
            ],
            "bitstream": input,
        }

    def decode(
        self,
        input: Dict,
        file_prefix: str = "",
    ):
        del file_prefix
        print("Decode")
        for layer_name, layer_data in input["data"].items():
            print(layer_name)
            print(layer_data.shape)
        return input
