
import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from src.evaluation import STOREBASEPATH
import time

def get_model_size(model):
    param_size = 0
    n_params = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        n_params += param.nelement()
    buffer_size = 0
    n_buffers = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        n_buffers += buffer.nelement()

    print('n_params:', n_params)
    print('n_buffers:', n_buffers)
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

@torch.no_grad()
def evaluate(dataset, n_imgs, model_refine=None, path=STOREBASEPATH+'/05_experiments/'):
    results = {}

    assert(dataset.featurizer is not None)
    # measure the runtime and memory usage
    # parameter sizes of dataset.featurizer model:
    models = dataset.featurizer.get_models()
    sizes = [get_model_size(model) for model in models]
    results["size featurizer (MB)"] = sum(sizes)

    if model_refine!=None:
        results["size refiner (MB)"] = get_model_size(model_refine)
    # measure the runtime 
    i=0
    time_sum = 0
    for cat in tqdm(dataset.all_cats):
        dataset.init_kps_cat(cat)
        for idx_ in range(len(dataset)):
            if i == n_imgs:
                break

            img = dataset._get_img(idx_)

            starttime = time.time()
            ft_orig = dataset.featurizer.forward(img,
                                    category=cat,
                                    **dataset.featurizer_kwargs)
            if model_refine!=None:
                ft_new = model_refine(ft_orig)
            endtime = time.time()
            time_sum += endtime - starttime
            i+=1

    endtime = time.time()
    results["runtime"] = time_sum/i *1000 # in ms
    return results
