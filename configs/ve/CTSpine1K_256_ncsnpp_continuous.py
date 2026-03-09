from configs.ve.AAPM_256_ncsnpp_continuous import get_config as get_default_configs



def get_config():
    config = get_default_configs()
    # data
    data = config.data
    data.image_size = 256
    
    data.dataset = "CTSpine1K"  # 这里必须与 datasets.py 中的判断字符串一致
    
    data.json = "/root/epfs/dataset_split.json"
    data.root = "/root/epfs/data_lmdb_seg/"
    data.seq = "CT"
    data.orientation = "SAG"
    data.hcp = False

    return config