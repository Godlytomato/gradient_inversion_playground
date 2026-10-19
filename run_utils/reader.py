#read yaml into dict
import yaml
#library for converting to dataclass
from dacite import from_dict
#####
from run_def import run_desc
from run_utils.bound_settings import normalize_bound_settings
from run_utils.svd_settings import normalize_min_svd_settings

#read run
#takes an input str converts to path and expects to return run object
def read_run(path: str) -> run_desc:
    with open(path, "r") as file:
        raw = yaml.safe_load(file)
    #converts the cit into the dataclass
    return normalize_min_svd_settings(normalize_bound_settings(from_dict(
        data_class=run_desc,
        data=raw,
    )))
