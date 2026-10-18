from reconstruction.optimization.schedulers.scheduler_def import (
    scheduler_def,
)

from reconstruction.optimization.schedulers.implementations.exponential import (
    exponential_scheduler,
)


scheduler_registry = {

    "exponential": scheduler_def(
        name="exponential",
        function=exponential_scheduler,
        requires=set(),
    ),
}