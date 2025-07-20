import wandb

def init_wandb(cfg, prefix=''):
    # start a new wandb run to track this script
    # check if feat model is in the cfg
    name = ''
    if 'featurizer' in cfg:
        name = name+cfg['featurizer']['model']
    if 'feat_refine' in cfg:
        name = name+'+'+cfg['feat_refine']['model']
    name = name+'+'+cfg['dataset']['name']
    if 'train' in prefix:
        if 'sup' in cfg['dataset']:
            name = name+cfg['dataset']['sup']
        if 'dataset2' in cfg:
            name = name+'+'+cfg['dataset2']['name']
            if 'sup' in cfg['dataset2']:
                name = name+cfg['dataset2']['sup']
        if 'dataset3' in cfg:
            name = name+'+'+cfg['dataset3']['name']
            if 'sup' in cfg['dataset3']:
                name = name+cfg['dataset3']['sup']
        if 'dataset4' in cfg:
            name = name+'+'+cfg['dataset4']['name']
            if 'sup' in cfg['dataset4']:
                name = name+cfg['dataset4']['sup']

    wandb.init(entity="<your-wandb-entity>",
                project="<your-wandb-project>",
                name=prefix+'  '+name,
                config=cfg,
                settings=wandb.Settings(code_dir="."))

def log_wandb(results):
    if wandb.run:
        wandb.log(results)  
    
def log_wandb_epoch(epoch):

    if wandb.run:
        results = {'epoch': epoch}
        wandb.log(results)

def log_wandb_cfg(cfg):
    # init wandb if not already
    if wandb.run:
        wandb.config.update(cfg, allow_val_change=True)

def log_wandb_ram_usage():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    other = psutil.virtual_memory()
    
    if wandb.run:
        wandb.log({ 'ram_usage': process.memory_info().rss/1024/1024
                    , 'ram_total': other.total/1024/1024
                    , 'ram_free': other.available/1024/1024
                    , 'ram_percent': other.percent})
    print('ram_usage: ', process.memory_info().rss/1024/1024, 'MB')
    print('ram_total: ', other.total/1024/1024, 'MB')
    
def finish_wandb():
    if wandb.run:
        wandb.finish()