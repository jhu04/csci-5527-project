import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import matplotlib.transforms as mtransforms
from tqdm import tqdm
def plot_by_epoch(arrs,labels,y,title,fn,r_value=None):
    # Generating a colormap to differentiate the arrays with unique colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(arrs)))

    plt.figure(figsize=(10, 6))
    for i, (array, color) in enumerate(zip(arrs, colors)):
        plt.plot(array, label=labels[i], color=color)
    if r_value is not None:
        plt.axhline(y=r, color='r', linestyle='--', label='r')

    #plt.title('Plot of 10 Arrays with Unique Colors')
    plt.xlabel('Epoch')
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(fn)
    plt.close()

def set_tick_labels_bold(ax, fontsize='large'):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)

def highlight_ytick(ax, y_value,fig,color='lightcoral', fontsize='large'):
    # Ensure y_value is in the existing y-ticks, if not, add it
    yticks = list(ax.get_yticks())
    if y_value not in yticks:
        yticks.append(y_value)
        yticks.sort()  # Sort the list to keep ticks in order
        ax.set_yticks(yticks)
    
    # Set y-tick labels, making the specified y_value label colored and bold
    yticklabels = [f'{label:.2f}' if label != y_value else f'${{\\mathbf{{{y_value}}}}}$' for label in yticks]
    ax.set_yticklabels(yticklabels)
    
    # Set y-tick label color and font properties
    for label in ax.get_yticklabels():
        if label.get_text() == f'${{\\mathbf{{{y_value}}}}}$':
            dx = 0
            dy = 0.2
            offset = mtransforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            label.set_color(color)
            #label.set_fontweight('bold')
            #label.set_verticalalignment('bottom') 
            label.set_transform(label.get_transform() + offset)
            #label.set_fontsize(fontsize)
        else:
            #label.set_fontweight('bold')
            label.set_fontsize('large')
        #label.set_fontsize(30)


def plot_result(data,r,x_names,y_names,fn):
    # Setting the size of each image and the grid dimensions
    
    #y_names = ['Feasibility', 'Objective','Accuracy Disparity' , 'Objective Accuracy']
    labels = ['train','test']
    #print(data.shape)
    #rint

    grid_dim = (len(y_names),len(x_names))
    subplot_hsize = 8 # size for each subplot
    subplot_vsize = 8.5
    colors = ['#1f77b4','#ff7f0e','#2ca02c']
    line_style = 'solid'
    line_width = 3

    legend_font_size = 48
    # Generating random images
    #random_images = np.random.rand(grid_dim[0], grid_dim[1], image_size[0], image_size[1])
    legend_dict = {}
   
    fig_hsize = grid_dim[1] * subplot_hsize
    fig_vsize = grid_dim[0] * subplot_vsize
    fig, axes = plt.subplots(grid_dim[0], grid_dim[1],sharex = 'col',sharey='row', figsize=(fig_hsize, fig_vsize))  # Keeping the figure square
    plt.tight_layout() 
    fig.subplots_adjust(top = 0.94,bottom=0.155,left=0.042,right=0.99) #,hspace=0.3, wspace=0.1
    for i in range(grid_dim[0]):
        for j in range(grid_dim[1]):
            if j == 0:
                top = axes[i,0].get_position().y0
                bottom = axes[i,0].get_position().y1
                center_y = (top + bottom)/2
                fig.text(0.01,center_y , y_names[i], ha='center', va='center', fontsize=48,rotation='vertical')
            if i == 0:
                left = axes[0,j].get_position().x0
                right = axes[0,j].get_position().x1
                center_x = (left+right)/2
                if 'l2' in x_names[j]:
                    fig.text(center_x, 0.97, r'$\ell_2$-Penalty', ha='center', va='center', fontsize=48)
                else:
                    fig.text(center_x, 0.97, x_names[j], ha='center', va='center', fontsize=48)
            if i == grid_dim[0] - 1:
                axes[i,j].set_xlabel('Epochs',fontsize=32)
            # now we plot the figure
            #print(f'{i},{j}')
            for k in range(len(data[i][j])):
                #print(f'{i},{j},{k}')
                arr = data[i][j][k]
                axes[i,j].plot(arr[:500], label=labels[k], color=colors[k],lw=line_width,linestyle = line_style)
            if y_names[i] == 'Feasibility':
                axes[i,j].axhline(y=r, color='r', linestyle='--', label='$\epsilon$')
                axes[i,j].text(x=(len(arr[:500])-1)/2, y=r, s='feasible if below the line', 
                               color='red', va='bottom', ha='center',fontsize=32,alpha=0.5,fontweight='bold')
                if j == 0:
                    highlight_ytick(axes[i,j],r,fig)
            handles, labels = axes[i, j].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_dict:
                    legend_dict[label] = handle
            #set_tick_labels_bold(axes[i,j])
    fig.legend(legend_dict.values(), legend_dict.keys(), loc='lower center', ncol=len(legend_dict.keys()), bbox_to_anchor=(0.5, 0),fontsize=legend_font_size)
    x_line = (axes[0,grid_dim[1]-2].get_position().x1+axes[0,grid_dim[1]-1].get_position().x0)/2
    for ax in axes.flat:
        ax.tick_params(axis='x', labelsize=30, which='both')
        ax.tick_params(axis='y', labelsize=30, which='both')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.label_outer()
    fig.add_artist(plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linestyle='--', linewidth=2))

    
    plt.savefig(fn)
    plt.close()

if __name__ == '__main__':
    base_folder = "/home/jusun/zhan7867/Deep_Learning_NTR_CST/exp/"
    plot_base_folder = '/home/jusun/zhan7867/Deep_Learning_NTR_CST/exp/figures'
    os.makedirs(plot_base_folder,exist_ok=True)
    paper_plot_base_folder = os.path.join(plot_base_folder,'paper')
    os.makedirs(paper_plot_base_folder,exist_ok=True)


    rho_ls = [10.0,7.5, 5.5, 2.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]
    optimizer_names = [f'penalty_{rho}' for rho in rho_ls]
    CCE = 50.0
    RF = 2.0
    num_repeats=1
    BN = 'No'
    data_folders = {opt_name:f'PENALTY_adult_self_cleaned_RHO_{rho}_CHECK_CONSTR_EVERY_{CCE}_RHO_FACTOR_{RF}_PT_l2' \
    #data_folders = {opt_name:f'PyGRANSO_PENALTY_adult_self_cleaned_RHO_{rho}'                
                    for rho, opt_name in zip(rho_ls,optimizer_names)}
    
    if BN == 'No':
        for opn in optimizer_names:
            data_folders[opn] = data_folders[opn] + '_BN_False'
    r_values = [0.005]#, 0.001,0.01]   #,0.3 #0.1,0.3,0.01
    file_names = [f"{i:06}" for i in range(num_repeats)]






    for r in tqdm(r_values): 
        plot_folder = os.path.join(plot_base_folder,f'r_{r}')
        paper_plot_folder = os.path.join(paper_plot_base_folder,f'r_{r}')
        os.makedirs(plot_folder,exist_ok=True)
        os.makedirs(paper_plot_folder,exist_ok=True)
        folder = {}
        paper_folder = {}
        
        data = {name:[] for name in optimizer_names}
        for file in file_names:
            for opt_name in optimizer_names:
                data[opt_name].append(np.load(os.path.join(base_folder,f'r_{r}',data_folders[opt_name],file + '.npz')))

        y_ls = ['diff','f','accuracy_diff','accuracy']
        y_name_ls = ['Feasibility','Objective','Accuracy Disparity','Objective Accuracy']
        init_id = 0
        labels = ['train','test']
        plot_large_data_arr = []
        for idx in range(len(file_names)):
            for y in y_ls:
                y_arr = []
                for opt_name in optimizer_names:
                    if y == 'accuracy_diff':
                        opt_data = data[opt_name][idx]
                        arr = [
                            np.abs(opt_data['accuracy0']-opt_data['accuracy1']),
                            np.abs(opt_data['accuracy0_test']-opt_data['accuracy1_test'])
                        ]
                        y_arr.append(arr)
                    else:
                        opt_data = data[opt_name][idx]
                        arr = [opt_data[y],opt_data[y+'_test']]
                        y_arr.append(arr)
                plot_large_data_arr.append(y_arr)
            plot_result(plot_large_data_arr,r,optimizer_names,y_name_ls,os.path.join(paper_plot_folder,f'penalties.png'))
    

    ################### Second Plot #####################
    RHO = [0.75,0.75]
    optimizer_names = ['fairlearn','tfco','penalty_q','penalty','pygranso','plain']
    #optimizer_names = ['tfco','penalty_pygranso','pygranso','plain']
    x_names = ['Fairlearn','TFCO', f'Quadratic Penalty', f'Exact Penalty', 'PyGRANSO','Unconstrained']
    #x_names = ['TFCO',f'Penalty + PyGRANSO', 'PyGRANSO','Unconstrained']
    CCE = 50.0
    RF = 2.0
    BN = 'No'
    num_repeats=1
    data_folders = {
        'fairlearn':'fairlearn',
        'pygranso':'PyGRANSO_adult_self_cleaned',
        'tfco':'TFCO_adult_self_cleaned',
        'penalty_pygranso':f'PyGRANSO_PENALTY_adult_self_cleaned_RHO_{RHO[0]}',
        'penalty_q':f'PENALTY_adult_self_cleaned_RHO_{RHO[1]}_CHECK_CONSTR_EVERY_{CCE}_RHO_FACTOR_{RF}_PT_l2',
        'penalty':f'PENALTY_adult_self_cleaned_RHO_{RHO[0]}_CHECK_CONSTR_EVERY_{CCE}_RHO_FACTOR_{RF}_PT_EXACT',
        'plain':'PLAIN_adult_self_cleaned'
    }

    
    if BN == 'No':
        for opn in optimizer_names:
            data_folders[opn] = data_folders[opn] + '_BN_False'
    r_values = [0.005]#, 0.001,0.01]   #,0.3 #0.1,0.3,0.01
    file_names = [f"{i:06}" for i in range(num_repeats)]
    #t = 3
    #file_names = [f'{t:06}']
    


    for r in tqdm(r_values): 
        plot_folder = os.path.join(plot_base_folder,f'r_{r}')
        paper_plot_folder = os.path.join(paper_plot_base_folder,f'r_{r}')
        os.makedirs(plot_folder,exist_ok=True)
        os.makedirs(paper_plot_folder,exist_ok=True)
        # folder = {}
        # paper_folder = {}
        # for name in optimizer_names:
        #     dir = os.path.join(plot_folder,name)
        #     folder[name] = dir
        #     os.makedirs(dir,exist_ok=True)
        #     dir1 = os.path.join(paper_plot_folder,name)
        #     paper_folder[name] = dir1
        #     os.makedirs(dir1,exist_ok=True)
        
        data = {name:[] for name in optimizer_names}
        for file in file_names:
            for opt_name in optimizer_names:
                data[opt_name].append(np.load(os.path.join(base_folder,f'r_{r}',data_folders[opt_name],file + '.npz')))
        # y_ls = ['accuracy','accuracy_test','f','f_test','diff','diff_test']
        # y_name_ls = ['accuracy','accuracy_test','obj','obj_test','constr','constr_test']
        # for y,y_name in zip(y_ls,y_name_ls):
        #     for opt_name in optimizer_names:
        #         arr = [opt_data[y] for opt_data in data[opt_name]]
        #         labels = ['init_'+fn for fn in file_names]
        #         title = f'{opt_name} ({y_name},r={r})'
                
        #         filename = os.path.join(folder[opt_name],f'{y_name}.png')
        #         if 'diff' in y:
        #             plot_by_epoch(arr,labels,y,title,filename,r)
        #         else:    
        #             plot_by_epoch(arr,labels,y,title,filename)
        
        # # make a separate plot for the accuracy_diff
        # for opt_name in optimizer_names:
        #     arr = [np.abs(opt_data['accuracy0']-opt_data['accuracy1']) for opt_data in data[opt_name]]
        #     label = [ 'init_'+fn for fn in file_names]
        #     title = f'{opt_name} (accuracy_diff, r={r})'
        #     filename = os.path.join(folder[opt_name],'accuracy_diff.png')
        #     plot_by_epoch(arr,labels,'accuracy_diff',title,filename)
        #     arr = [np.abs(opt_data['accuracy0_test']-opt_data['accuracy1_test']) for opt_data in data[opt_name]]
        #     label = [ 'init_'+fn for fn in file_names]
        #     title = f'{opt_name} (accuracy_diff_test, r={r})'
        #     filename = os.path.join(folder[opt_name],'accuracy_diff_test.png')
        #     plot_by_epoch(arr,labels,'accuracy_diff_test',title,filename)

        # now we plot only one init with train and test on the same plot
        y_ls = ['diff','f','accuracy_diff','accuracy']
        y_name_ls = ['Feasibility','Objective']#,'Accuracy Disparity','Objective Accuracy']
        init_id = 0
        labels = ['train','test']
        # for y,y_name in zip(y_ls,y_name_ls):
        #     for opt_name in optimizer_names:
        #         if y == 'accuracy_diff':
        #             opt_data = data[opt_name][init_id]
        #             arr = [
        #                 np.abs(opt_data['accuracy0']-opt_data['accuracy1']),
        #                 np.abs(opt_data['accuracy0_test']-opt_data['accuracy1_test'])
        #                 ]
        #         else:
        #             opt_data = data[opt_name][init_id]
        #             arr = [opt_data[y],opt_data[y+'_test']]
        #         title = f'{opt_name}(r = {r})'
        #         filename = os.path.join(paper_folder[opt_name],f'Paper_{y}.png')
        #         if y == 'diff':
        #             plot_by_epoch(arr,labels,y_name,title,filename,r)
        #         else:    
        #             plot_by_epoch(arr,labels,y_name,title,filename)
        
        plot_large_data_arr = []
        for idx in range(len(file_names)):
            for y in y_ls:
                y_arr = []
                for opt_name in optimizer_names:
                    if opt_name == 'fairlearn':
                        opt_data = data[opt_name][idx]
                        arr = [opt_data[y][:,0],opt_data[y][:,1]]
                        y_arr .append(arr)
                    else:
                        if y == 'accuracy_diff':

                            opt_data = data[opt_name][idx]
                            arr = [
                                np.abs(opt_data['accuracy0']-opt_data['accuracy1']),
                                np.abs(opt_data['accuracy0_test']-opt_data['accuracy1_test'])
                            ]
                            y_arr.append(arr)
                        else:
                            opt_data = data[opt_name][idx]
                            arr = [opt_data[y],opt_data[y+'_test']]
                            y_arr.append(arr)
                plot_large_data_arr.append(y_arr)
            plot_result(plot_large_data_arr,r,x_names,y_name_ls,os.path.join(paper_plot_folder,f'r_{r}_init_{idx}.png'))
            #plot_result(plot_large_data_arr[:2],r,x_names,['Feasibility', 'Objective'],os.path.join(paper_plot_folder,f'r_{r}_init_{idx}.png'))
            #plot_result(plot_large_data_arr[2:],r,x_names,['Accuracy Disparity', 'Objective Accuracy'],os.path.join(paper_plot_folder,f'r_{r}_init{idx}_acc.png'))
    
    