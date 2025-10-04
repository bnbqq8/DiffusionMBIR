from configs.ve.AAPM_256_ncsnpp_continuous import get_config as get_default_configs


def get_config():
    config = get_default_configs()
    # data
    data = config.data
    data.dataset = "IXI"
    data.json = "../IXI_diffusion/dataset_split.json"
    data.seq = "T1"
    data.orientation = "AX"  # AX, SA, or COR

    return config
