from config.config_template import ConfigTemplate


def build_norm(config: ConfigTemplate):
    if config.norm_name == "LayerNorm":
        from model.modules.norm.layer_norm import LayerNorm
        return LayerNorm(config)
    elif config.norm_name == "RMSNorm":
        from model.modules.norm.rms_norm import RMSNorm
        return RMSNorm(config)
    else:
        raise Exception("Unexpected norm_name")
