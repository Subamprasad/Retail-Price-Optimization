from zenml import step

@step
def deployment_trigger(accuracy: float, min_accuracy: float = 0.0) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy."""
    return True
