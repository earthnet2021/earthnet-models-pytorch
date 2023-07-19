import sys
from torchmetrics import Metric
import torch


class RootMeanSquaredError(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        lc_min: int,
        lc_max: int,
        context_length: int,
        target_length: int,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            # Advanced metric settings
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("current_scores", default=torch.tensor(0), dist_reduce_fx=None)

        self.context_length = context_length
        self.target_length = target_length

        self.lc_min = lc_min
        self.lc_max = lc_max
        print(
            f"Using Masked RootMeanSquaredError metric Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
        )

    def update(self, preds, targs):
        """Any code needed to update the state given any inputs to the metric."""

        # Targets
        targets = targs["dynamic"][0][
            :, self.context_length : self.context_length + self.target_length, 0, ...
        ].unsqueeze(2)

        # Masks on the non vegetation pixels
        # Dynamic cloud mask available
        if len(targs["dynamic_mask"]) > 0:
            s2_mask = (
                (
                    targs["dynamic_mask"][0][
                        :,
                        self.context_length : self.context_length + self.target_length,
                        ...,
                    ]
                    < 1.0
                )
                .bool()
                .type_as(preds)
            )

        # Landcover mask
        lc = targs["landcover"]
        lc_mask = (
            ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
            .type_as(s2_mask)
            .unsqueeze(1)
            .repeat(1, preds.shape[1], 1, 1, 1)
        )

        mask = s2_mask * lc_mask 

        # MSE computation
        sum_squared_error = torch.pow((preds - targets) * mask, 2).sum((1, 2, 3, 4))
        n_obs = (mask == 1).sum((1, 2, 3, 4)) #sum of pixel with vegetation

        # Update the states variables
        self.current_scores = sum_squared_error / (n_obs + 1e-8)
        self.sum_squared_error += sum_squared_error.sum()
        self.total += n_obs.sum()

    def compute(self):
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.total)}

    def compute_batch(self, targs):
        """
        Computes a final value for each sample of a batch over the state of the metric.
        """
        cubenames = targs["cubename"]
        return [
            {
                "name": cubenames[i],
                "rmse": self.current_scores[i],
            }  # "rmse" is Legacy, update logging before
            for i in range(len(cubenames))
        ]
