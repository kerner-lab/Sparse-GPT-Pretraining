def to_cpu_recursive(item):
    if hasattr(item, "cpu"):
        return item.cpu()
    elif isinstance(item, dict):
        return {k: to_cpu_recursive(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [to_cpu_recursive(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(to_cpu_recursive(v) for v in item)
    else:
        return item
