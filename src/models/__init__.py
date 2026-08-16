from .resunet import ResUNet
from .siglip2_unet import Siglip2UNet
from .siglip_so400m_unet import SiglipSO400MUNet

def build_model(cfg: dict):
    m = cfg["model"]
    name = m["name"]
    if name == "resunet":
        return ResUNet(
            in_channels=int(m.get("in_channels", 3)),
            out_channels=int(m.get("out_channels", 1)),
            base_channels=int(m.get("base_channels", 32)),
        )
    if name == "siglip2_unet":
        return Siglip2UNet(
            checkpoint=m["checkpoint"],
            feature_layers=list(m.get("feature_layers", [3, 6, 9, 12])),
            decoder_channels=int(m.get("decoder_channels", 256)),
            out_channels=int(m.get("out_channels", 1)),
            train_mode=m.get("train_mode", "full"),
            partial_last_n=int(m.get("partial_last_n", 4)),
            load_base_pretrained=bool(m.get("load_base_pretrained", True)),
        )
    if name == "siglip_so400m_unet":
        return SiglipSO400MUNet(
            checkpoint=m.get(
                "checkpoint", "google/siglip2-so400m-patch14-384"
            ),
            classification_checkpoint=m.get("classification_checkpoint"),
            classification_state=m.get("classification_state", "auto"),
            strict_classification_init=bool(
                m.get("strict_classification_init", True)
            ),
            load_base_pretrained=bool(m.get("load_base_pretrained", True)),
            feature_layers=list(m.get("feature_layers", [6, 12, 20, 27])),
            decoder_channels=int(m.get("decoder_channels", 256)),
            out_channels=int(m.get("out_channels", 1)),
            train_mode=m.get("train_mode", "frozen"),
            partial_last_n=int(m.get("partial_last_n", 6)),
        )
    raise ValueError(f"Unknown model: {name}")
