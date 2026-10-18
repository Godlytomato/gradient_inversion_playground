def get_device(context):
    return next(context.model.parameters()).device