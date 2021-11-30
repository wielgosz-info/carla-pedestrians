import argparse


class MinMaxAction(argparse.Action):
    def __init__(self, option_strings, dest, minimum=None, maximum=None, **kwargs) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, parser, namespace, values, option_string=None):
        if self.minimum is not None and values < self.minimum:
            raise parser.error(
                f"{self.dest} must be greater than or equal to {self.minimum}")
        if self.maximum is not None and values > self.maximum:
            raise parser.error(
                f"{self.dest} must be less than or equal to {self.maximum}")

        setattr(namespace, self.dest, values)
