import os

class TrainingRunReport:
    """
    A utility class to track training and validation metrics and generate a Markdown report.
    """

    def __init__(self, model_name, output_path):
        """
        Initialize the report generator.

        Args:
            model_name (str): Name of the model being trained.
            output_path (str): Path to save the Markdown report.
        """
        self.model_name = model_name
        self.output_path = output_path
        self.epoch_logs = []

    def log_epoch(self, epoch, train_loss, val_loss=None):
        """
        Log metrics for a single epoch.

        Args:
            epoch (int): Epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (float, optional): Validation loss for the epoch. Defaults to None.
        """
        self.epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    def save_report(self):
        """
        Generate and save the Markdown report.
        """
        report_lines = [
            f"# Training Report for {self.model_name}",
            "",
            "## Epoch-wise Metrics",
            "| Epoch | Training Loss | Validation Loss |",
            "|-------|---------------|-----------------|"
        ]

        for log in self.epoch_logs:
            val_loss = f"{log['val_loss']:.4f}" if log["val_loss"] is not None else "N/A"
            report_lines.append(f"| {log['epoch']} | {log['train_loss']:.4f} | {val_loss} |")

        report_content = "\n".join(report_lines)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(report_content)
