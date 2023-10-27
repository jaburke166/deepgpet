import numpy as np
import skimage as sk
import os
import pandas as pd
import pickle
import PIL.Image as Image
import matplotlib.pyplot as plt


##############################################################################################################
                                    # LOADING AND PLOTTING
##############################################################################################################


def load_img(path, ycutoff=0, xcutoff=0):
    '''
    Helper function to load in image and crop
    '''
    img = np.array(Image.open(path))[ycutoff:, xcutoff:]/255.0
    ndim = img.ndim
    M, N = img.shape[:2]
    pad_M = (32 - M%32) % 32
    pad_N = (32 - N%32) % 32

    # Assuming third color channel is last axis
    if ndim == 2:
        return np.pad(img, ((0, pad_M), (0, pad_N)))
    else: 
        return np.pad(img, ((0, pad_M), (0, pad_N), (0,0)))


def plot_img(img_data, traces=None, cmap=None, save_path=None, fname=None, sidebyside=False, rnfl=False):
    '''
    Helper function to plot the result - plot the image, traces, colourmap, etc.
    '''
    img = img_data.copy().astype(np.float64)
    img -= img.min()
    img /= img.max()
    M, N = img.shape
    
    if rnfl:
        figsize=(15,6)
    else:
        figsize=(6,6)

    if sidebyside:
        figsize = (2*figsize[0], figsize[1])
    
    if sidebyside:
        fig, (ax0, ax) = plt.subplots(1,2,figsize=figsize)
        ax0.imshow(img, cmap="gray", zorder=1)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(img, cmap="gray", zorder=1)
    fontsize=16
    if traces is not None:
        if len(traces) == 2:
            for tr in traces:
                 ax.plot(tr[:,0], tr[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)
        else:
            ax.plot(traces[:,0], traces[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)

    if cmap is not None:
        cmap_data = cmap.copy().astype(np.float64)
        cmap_data -= cmap_data.min()
        cmap_data /= cmap_data.max()
        ax.imshow(cmap_data, alpha=0.5, zorder=2)
    if fname is not None:
        ax.set_title(fname, fontsize=15)
            
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    ax.axis([0, N-1, M-1, 0])
    fig.tight_layout(pad = 0)
    if save_path is not None and fname is not None:
        ax.set_title(None)
        fig.savefig(os.path.join(save_path, f"{fname}.png"), bbox_inches="tight")


def generate_imgmask(mask, thresh=None, cmap=0):
    '''
    Given a prediction mask Returns a plottable mask
    '''
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot



##############################################################################################################
                                    # POST-PROCESS IMAGE SEGMENTATION
##############################################################################################################


def extract_bounds(mask):
    '''
    Given a binary mask, return the top and bottom boundaries, 
    assuming the segmentation is fully-connected.
    '''
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T
    
    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:,0]))
    
    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1,np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0,np.newaxis],
                                 where_ones[sort_idxs+1].squeeze()], axis=0)
    
    return (top_bounds, bot_bounds)


def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = sk.measure.label(binmask)                       
    regions = sk.measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


def interp_trace(traces, align=True):
    '''
    Quick helper function to make sure every trace is evaluated 
    across every x-value that it's length covers.
    '''
    new_traces = []
    for i in range(2):
        tr = traces[i]  
        min_x, max_x = (tr[:,0].min(), tr[:,0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:,0], tr[:,1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1,1), y_interp.reshape(-1,1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx=0
        top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
        common_st_idx = max(top[0,h_idx], bot[0,h_idx])
        common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
        shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
        shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)


def crop_trace(traces, check_idx=60, offset=10):
    '''
    Crop trace left and right by searching check_idx either side to ensure
    that there is at least an offset pixel height
    '''
    # Align traces horizontally
    h_idx = 0
    top,bot = traces
    top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
    common_st_idx = max(top[0,h_idx], bot[0,h_idx])
    common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
    shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
    shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]

    # Now check to make sure at least an offset pixel height difference
    height_idx = shifted_bot[:,1] - shifted_top[:,1]
    left_idx = np.where(height_idx[:check_idx] >= offset)[0]
    right_idx = np.where(height_idx[-check_idx:] >= offset)[0]-check_idx

    # Reconstruct bounds
    new_top = np.concatenate([top[left_idx], top[check_idx:-check_idx], top[right_idx]], axis=0)
    new_bot = np.concatenate([bot[left_idx], bot[check_idx:-check_idx], bot[right_idx]], axis=0)
    new_trace = np.asarray(interp_trace(np.concatenate([new_top[np.newaxis], new_bot[np.newaxis]], axis=0)))

    return new_trace


def get_trace(pred_mask, threshold=0.5):
    '''
    Helper function to extract traces from a prediction mask. 
    '''
    binmask = (pred_mask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = interp_trace(traces, align=False)
    return traces
    

def rebuild_mask(traces, img_shape=None):
    '''
    Rebuild binary mask from choroid traces
    '''
    # Work out extremal coordinates of traces
    top_chor, bot_chor = traces
    common_st_idx = np.maximum(top_chor[0,0], bot_chor[0,0])
    common_en_idx = np.minimum(top_chor[-1,0], bot_chor[-1,0])
    top_idx = top_chor[:,1].min()
    bot_idx = bot_chor[:,1].max()

    if img_shape is not None:
        binmask = np.zeros(img_shape)
    else:
        binmask = np.zeros((bot_idx+100, common_en_idx+100))

    for i in range(common_st_idx, common_en_idx):
        top_i = top_chor[i-common_st_idx,1]
        bot_i = bot_chor[i-common_st_idx,1]
        binmask[top_i:bot_i,i] = 1

    return binmask