import warnings

from tqdm.std import TqdmExperimentalWarning

warnings.filterwarnings(
    "ignore",
    message=r"Using `tqdm\.autonotebook\.tqdm` in notebook mode\..*",
    category=TqdmExperimentalWarning,
)
