from torchmetrics import Metric
import torch


class RootMeanSquaredError(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        context_length: int,
        target_length: int,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        """
        Initialize the RootMeanSquaredError metric.

        Parameters
        ----------
        lc_min : int
            Minimum value bound for the landcover mask.
        lc_max : int
            Maximum value bound for the landcover mask.
        context_length : int
            Length of the context for predictions.
        target_length : int
            Length of the target sequence for predictions.
        compute_on_step : bool, optional
            If True, compute the metric on each forward call (default is False).
        dist_sync_on_step : bool, optional
            If True, synchronize metric computation across devices in a distributed environment (default is False).
        process_group : optional
            Process group for distributed training (default is None).
        dist_sync_fn : optional
            Function for synchronizing metric computation across devices (default is None).
        """
        super().__init__(
            # Advanced metric settings
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        # State variables for the metric
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("current_scores", default=torch.tensor(0), dist_reduce_fx=None)

        self.context_length = context_length
        self.target_length = target_length

        print(f"Using Masked RootMeanSquaredError metric Loss.")

    def update(self, preds, targs):
        """
        Update the state variables of the metric given the input predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted tensor from the model.
        targs : dict
            Dictionary containing the target data.
        """

        # Targets for the RMSE computation
        targets = targs["dynamic"][0][
            :, self.context_length : self.context_length + self.target_length, 0, ...
        ].unsqueeze(2)

        # Masks on the non-vegetation pixels
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
        lc_mask = targs["landcover_mask"]

        if len(lc_mask.shape) < 5:  # spatial landcover mask
            lc_mask = lc_mask.unsqueeze(1).repeat(1, preds.shape[1], 1, 1, 1)
        else:  # spato-temporal landcover mask
            lc_mask = lc_mask[
                :, self.context_length : self.context_length + self.target_length, ...
            ]

        mask = s2_mask * lc_mask

        # Compute the Mean Squared Error (MSE) for the masked regions
        sum_squared_error = torch.pow((preds - targets) * mask, 2).sum((1, 2, 3, 4))
        n_obs = (mask == 1).sum((1, 2, 3, 4))  # sum of pixel with vegetation

        # Update the states variables
        self.current_scores = torch.sqrt(sum_squared_error / (n_obs + 1e-8))
        self.total += (sum_squared_error / (n_obs + 1e-8)).sum()
        self.n_samples += n_obs.shape[0]

    def compute(self):
        """
        Compute the final Root Mean Squared Error (RMSE) over the state of the metric.

        Returns
        -------
        dict
            Dictionary containing the computed RMSE for vegetation pixels.
        """
        return {"RMSE_Veg": torch.sqrt(self.total / (self.n_samples))}

    def compute_batch(self, targs):
        """
        Compute the final RMSE for each sample of a batch over the state of the metric.

        Parameters
        ----------
        targs : dict
            Dictionary containing the target data.

        Returns
        -------
        list
            List of dictionaries containing RMSE scores for each sample in the batch.
        """
        cubenames = targs["cubename"]
        return [
            {
                "name": cubenames[i],
                "rmse": self.current_scores[i],
            }
            for i in range(len(cubenames))
        ]
