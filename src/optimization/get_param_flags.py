def get_param_flags(unwrapped_model, p):
    param_flags = {"decay": False, "8bit": False}
    for module in unwrapped_model.modules():
        if hasattr(module, "params_decay"):
            if any(p is q for q in module.params_decay):
                param_flags["decay"] = True
        if hasattr(module, "params_8bit"):
            if any(p is q for q in module.params_8bit):
                param_flags["8bit"] = True
    return param_flags
