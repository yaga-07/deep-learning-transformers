import logging
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "\033[1;34m[%(asctime)s]\033[0m \033[1;32m%(levelname)s\033[0m - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class RichLogger:
    def __init__(self, name="rich_logger", level=logging.DEBUG):
        self.console = Console()
        
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(name)

    def new_run(self, message="New run", style="bold yellow"):
        """Display a header panel for a new run."""
        self.console.print(Panel.fit(message, style=style))

    def step(self, step_number, description, style="bold yellow"):
        """Display a step indicator."""
        self.console.print(f"[{style}]Step {step_number}:[/] {description}")

    def error(self, message, url=None, request_id=None):
        """Display an error in styled format."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")
        if url:
            self.console.print(f"[blue underline]{url}[/blue underline]")
        if request_id:
            self.console.print(f"[bold red]Request ID:[/bold red] {request_id}")

    def info(self, message):
        """Log an info-level message."""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug-level message."""
        self.logger.debug(message)

    def warning(self, message):
        """Log a warning-level message."""
        self.logger.warning(message)

    def critical(self, message):
        """Log a critical-level message."""
        self.logger.critical(message)

    def exception(self, message):
        """Log an exception with traceback."""
        self.logger.exception(message)

