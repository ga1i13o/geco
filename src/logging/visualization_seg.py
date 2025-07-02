import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import wandb

COLOURS = [
(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
(255, 255, 0), (0, 0, 255), (0, 128, 255), (128, 0, 255), (0, 128, 0),
(128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
]

def plot_assignment(assignment_dict, list_of_kp_names=None, path=None, wandb_suffix='', plot_cmap=False, limits=None):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    def plot_matrix(M_, title, limits):
        if 'single' in path and "cos_sim" in title:
            M = M_.clone()
            limits = [0, 1]
        elif 'single' in path and "diag" in path:
            M = M_.clone()
        else:
            M = M_.clone()/M_.sum()
        ratio = M.shape[1]/M.shape[0]
        plt.figure(figsize=(5*ratio,5))
        if limits is None:
            plt.imshow(M.detach().cpu().numpy(), cmap='jet')
        else:
            plt.imshow(M.detach().cpu().numpy(), cmap='jet', vmin=limits[0], vmax=limits[1])
        plt.yticks(())
        plt.xticks(())
        # title_suffix = 'max:%.4f' % M.max().item()
        # plt.title(title + ' - ' + title_suffix)
        plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
        if list_of_kp_names is not None:
            # x ticks on top
            plt.xticks(range(len(list_of_kp_names)), list_of_kp_names, rotation=90)
            plt.yticks(range(len(list_of_kp_names)), list_of_kp_names)
        # adjust the plot such that the heading and the colorbar are not cut off
        plt.subplots_adjust(top=0.78)
        plt.savefig(path+'/matrix_'+title + ".png")
        if wandb.run is not None:
            # log to wandb
            wandb.log({wandb_suffix+'_'+title : wandb.Image(path+'/matrix_'+title + ".png")})
        if plot_cmap:
            # plot the jet colormap
            plt.gca().set_visible(False)
            cax = plt.axes([0.0, 0.0, 1.0, 0.05])
            plt.colorbar(orientation="horizontal", cax=cax)
            plt.savefig(path+'/colourbar_'+title + ".png", bbox_inches='tight')

        plt.close('all')

    for key, value in assignment_dict.items():
        # if value.sum() > 0:
        # check for sparse matrix
        if hasattr(value, 'to_dense'):
            plot_matrix(value.to_dense(), key, limits)
        else:
            plot_matrix(value, key, limits)
