from typing import Final, List, Optional, Tuple

GH_CONTENT_URL: Final[str] = "https://raw.githubusercontent.com/"


class Config:
    BASE_URL: Final[str] = "http://sceneparsing.csail.mit.edu/model/pytorch/"
    GH_BASE_URL: Final[str] = (
        GH_CONTENT_URL + "CSAILVision/semantic-segmentation-pytorch/master/"
    )
    DECODER_FMT: Final[str] = "{}_decoder_epoch_{}.pth"
    ENCODER_FMT: Final[str] = "{}_encoder_epoch_{}.pth"
    CFG_URL_FMT: Final[str] = GH_BASE_URL + "config/{}.yaml"
    COLOR_MAT_URL: Final[str] = GH_BASE_URL + "data/color150.mat"
    LABEL_CSV_URL: Final[str] = GH_BASE_URL + "data/object150_info.csv"
    MODEL_INFOS: Final[List[Tuple[str, int, Optional[str]]]] = [
        ("ade20k-hrnetv2-c1", 30, "ade20k-hrnetv2"),
        (
            "ade20k-mobilenetv2dilated-c1_deepsup",
            20,
            "ade20k-mobilenetv2dilated-c1_deepsup",
        ),
        (
            "ade20k-resnet18dilated-c1_deepsup",
            20,
            None,
        ),
        (
            "ade20k-resnet18dilated-ppm_deepsup",
            20,
            "ade20k-resnet18dilated-ppm_deepsup",
        ),
        (
            "ade20k-resnet50-upernet",
            30,
            "ade20k-resnet50-upernet",
        ),
        (
            "ade20k-resnet50dilated-ppm_deepsup",
            20,
            "ade20k-resnet50dilated-ppm_deepsup",
        ),
        ("ade20k-resnet101-upernet", 50, "ade20k-resnet101-upernet"),
        (
            "ade20k-resnet101dilated-ppm_deepsup",
            25,
            "ade20k-resnet101dilated-ppm_deepsup",
        ),
    ]
    """Final[List[Tuple[str, int, Optional[str]]]]: the mit_semseg model info.

    List of tuples of (model_name, epochs, config_name (optional))"""

    DEFAULT_MODEL_NAME: Final[str] = "ade20k-resnet101dilated-ppm_deepsup"

    def __init__(self):
        self._model_names: List[str] = [x[0] for x in self.MODEL_INFOS]

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    def get_epoch(self, model_name: str) -> int:
        for model_info in self.MODEL_INFOS:
            if model_info[0] == model_name:
                return model_info[1]
        raise ValueError(f"Failed to find epoch value corresponding to {model_name}")

    def get_cfg_url(self, model_name: str) -> Optional[str]:
        for model_info in self.MODEL_INFOS:
            if model_info[0] == model_name:
                return self.CFG_URL_FMT.format(model_info[2]) if model_info[2] else None
        raise ValueError(
            f"Failed to find URL for configuration file corresponding to {model_name}"
        )

    def get_color_mat_url(self) -> str:
        return self.COLOR_MAT_URL

    def get_label_csv_url(self) -> str:
        return self.LABEL_CSV_URL
