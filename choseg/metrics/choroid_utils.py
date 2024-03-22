import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import logging
import pandas as pd
import pickle
import os
from PIL import Image
from scipy import interpolate as interp
from skimage import exposure
from skimage import measure as meas
from skimage import morphology as morph
from sklearn.linear_model import LinearRegression
from .pixel_gui import select_pixels


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


def curve_length(curve, scale=(11.49,3.87)):
    """
    Calculate the length (in microns) of a curve defined by a numpy array of coordinates.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Scale constants
    xum_per_pix, yum_per_pix = scale

    # Calculate difference between pairwise consecutive coordinates of curve
    diff = np.abs((curve[1:] - curve[:-1]).astype(np.float64))

    # Convert pixel difference to micron difference
    diff[:, 0] *= xum_per_pix
    diff[:, 1] *= yum_per_pix

    # Length is total euclidean distance between all pairwise-micron-movements
    length = np.sum(np.sqrt(np.sum((diff) ** 2, axis=1)))

    return length


def curve_location(curve, distance=2000, ref_idx=400, scale=(11.49,3.87), verbose=0, image_axis=False):
    """
    Given a curve, what two coordinates are *distance* microns away from some coordinate indexed by
    *ref_idx*.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Work out number of microns per unit pixel movement
    N = curve.shape[0]
    
    # Scale constants
    xum_per_pix, yum_per_pix = scale

    # If measuring along choroid axis
    if not image_axis:

        # Calculate difference between pairwise consecutive coordinates of curve
        diff_r = np.abs((curve[1 + ref_idx:] - curve[ref_idx:-1]).astype(np.float64))
        diff_l = np.abs((curve[::-1][1 + (N - ref_idx):] - curve[::-1][(N - ref_idx):-1]).astype(np.float64))
    
        # Convert pixel difference to micron difference
        diff_r[:, 0] *= xum_per_pix
        diff_r[:, 1] *= yum_per_pix
        diff_l[:, 0] *= xum_per_pix
        diff_l[:, 1] *= yum_per_pix
    
        # length per movement is euclidean distance between pairwise-micron-movements
        length_l = np.sqrt(np.sum((diff_l) ** 2, axis=1))
        cumsum_l = np.cumsum(length_l)
        length_r = np.sqrt(np.sum((diff_r) ** 2, axis=1))
        cumsum_r = np.cumsum(length_r)
    
        # Work out largest index in cumulative length sum where it is smaller than *distance*
        idx_l = ref_idx - np.argmin(cumsum_l < distance)
        idx_r = ref_idx + np.argmin(cumsum_r < distance)
        if (idx_l == ref_idx) and distance > 200:
            if verbose==1:
                logging.warning(f"""Segmentation not long enough for {distance}um left of fovea.
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None
        if (idx_r == ref_idx) and distance > 200:
            if verbose==1:
                logging.warning(f"""Segmentation not long enough for {distance}um right of fovea. 
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None
            
    # If measuring along image axis
    else:
        d_px = int(distance / xum_per_pix)
            
        if ((ref_idx-d_px) < 0) or ((d_px+ref_idx) > N):
            if verbose==1:
                logging.warning(f"""Segmentation not long enough for {distance}um right of fovea. 
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None
        idx_l, idx_r = ref_idx-d_px, ref_idx+d_px

    return idx_l, idx_r


def _check_offset(offset, offsets_lr, N_pts):
    '''
    Quick helper function to check if offset is too large, and deal with it if so
    '''
    (offset_l, offset_r) = offsets_lr
    if offset_l < 0:
        offset_l = 0
        logging.warning(f"Offset {offset} too far to the left, choosing index {offset_l}")
        
    if offset_r >= N_pts:
        offset_r = N_pts-1
        logging.warning(f"Offset {offset} too far to the right, choosing index {offset_r}")

    return offset_l, offset_r


def nearest_coord(trace, coord, offset=15, columnwise=False):
    """
    Given a coordinate, find the nearest coordinate along trace and return it.

    INPUTS:
    ------------------------
        trace (np.array) : Upper or lower choroid boundary

        coord (np.array) : Single xy-coordinate.

        offset (int) : Integer index to select either side of reference point
        along trace for deducing locally perpendicular CT measurement.

        columnwise (bool) : If flagged, nearest coordinate on the trace is
            the one at the same column index.

    RETURNS:
    ------------------------
        trace_refpt (np.array) : Point along trace closest to coord.

        offset_pts (np.array) : Points offset distance from trace_refpt for deducing
            local tangent line.
    """
    N_pts = trace.shape[0]

    # Work out closest coordinate on trace to coord
    if not columnwise:
        fovea_argmin = np.argmin(np.sum((trace - coord) ** 2, axis=1))
        trace_refpt = trace[fovea_argmin]
    else:
        fovea_argmin = coord[0] - trace[0,0]
        trace_refpt = trace[fovea_argmin]

    # Prevent error by choosing maximum offset, if offset is too large for given trace
    offset_l, offset_r = fovea_argmin - offset, fovea_argmin + offset
    offset_l, offset_r = _check_offset(offset, (offset_l, offset_r), N_pts)
    offset_pts = trace[[offset_l, offset_r]]
    
    return trace_refpt, offset_pts


def construct_line(p1, p2):
    """
    Construct straight line between two points p1 and p2.

    INPUTS:
    ------------------
        p1 (1d-array) : 2D pixel coordinate.

        p2 (1d-array) : 2D pixel coordinate.

    RETURNS:
    ------------------
        m, c (floats) : Gradient and intercept of straight line connection points p1 and p2.
    """
    # Measure difference between x- and y-coordinates of p1 and p2
    delta_x = (p2[0] - p1[0])
    delta_y = (p2[1] - p1[1])

    # Compute gradient and intercept
    try:
        assert delta_x != 0
        m = delta_y / delta_x
        c = p2[1] - m * p2[0]
    except AssertionError:
        m = np.inf
        c = np.inf

    return m, c


def generate_perp_line(pt1, pt2=None, N=None, ref_pt=None):
    """
    Linear model of tangent line centred at reference
    point along RPE-Choroid boundary, evaluate far enough 
    such that rotating by 90 degrees will ensure its
    intersection with Choroid-Sclera boundary
    """
    # Fit linear model at reference points along tangent
    if pt2 is None:
        X, y = pt1[:,0].reshape(-1,1), pt1[:,1]
    else:
        X, y = np.array([pt1[0], pt2[0]]).reshape(-1,1), np.array([pt1[1], pt2[1]])    
    output = LinearRegression().fit(X, y)

    # Evaluate across tangent, rotate at reference point and return perpendicular
    # linear model
    if N is not None and ref_pt is not None:
        
        ref_x, ref_y = ref_pt
        xtan_grid = np.array([ref_x, X[-1,0]+N])
        ytan_grid = output.predict(xtan_grid.reshape(-1,1)).astype(int)
        
        perp_x = (-(ytan_grid - ref_y) + ref_x).reshape(-1,)
        perp_y = (xtan_grid - ref_x + ref_y).reshape(-1,)
        output = (perp_x, perp_y)

        y_grid = np.arange(perp_y[0], perp_y[1])
        x_grid = np.interp(y_grid, perp_y, perp_x)
        output = (x_grid, y_grid)
        
    return output



def detect_orthogonal_chorscl(reference_pts, traces, offset=15, tol=2):
    """
    Given the lower choroid boundary and reference points along the upper boundary, detect which
    coordinates along the lower choroid boundary which intersect the perpendicular line drawn from
    a tangent line at these reference points.

    INPUTS:
    ------------------
        reference_pts (np.array) : Points along upper boundary to construct tangent and perpendicular lines at. 

        traces (2-tuple) : Tuple storing the upper and lower boundaries of the segmented chorod, in xy-space.

        offset (int) : Value either side of reference point to define tangential line.

        tol (int) : Threshold to detect any perpendicular lines from the upper to lower boundaries
        which are deviate away the lower boundary via Euclidean distance, i.e. it's likely these lines 
        divert away from the segmented region.
    """
    # Extract traces    
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

    # total number of candidate points at each reference point to compare with 
    # Choroid-Sclera boundary
    N = max([bot_chor[ref_x-chorscl_stx, 1] - ref_y for (ref_x, ref_y) in reference_pts])
    perps = []
    for ref_pt in reference_pts:
    
        # Work out local tangent line for each reference point
        # and rotate orthogonally
        ref_x, ref_y = ref_pt
        ref_xidx = ref_x - rpechor_stx
        #N = bot_chor[ref_x-chorscl_stx, 1] - ref_y
        tan_pt1, tan_pt2 = top_chor[[ref_xidx - offset, ref_xidx + offset]] 
        (perp_x, perp_y) = generate_perp_line(tan_pt1, tan_pt2, N, ref_pt)
        perps.append(np.array([perp_x, perp_y]))
    
    # Vectorised search for points along Choroid-Sclera boundary where orthogonal 
    # lines from RPE-Choroid intersect
    perps = np.array(perps)
    bot_cropped = bot_chor[(perps[:,0].astype(int)-chorscl_stx).clip(0, bot_chor.shape[0]-1)]
    bot_perps_residuals = np.transpose(perps, (0,2,1)) - bot_cropped
    bot_perps_distances = np.sqrt(((bot_perps_residuals)**2).sum(axis=-1))
    # print(bot_perps_distances.shape)
    # print(np.min(bot_perps_distances, axis=-1))
    endpoint_errors = np.min(bot_perps_distances, axis=-1) <= tol 
    chorscl_indexes = np.argmin(bot_perps_distances, axis=1)
    chorscl_pts = perps[np.arange(chorscl_indexes.shape[0]),:,chorscl_indexes].astype(int)

    return chorscl_pts.astype(float), reference_pts.astype(float), perps, endpoint_errors



##############################################################################################################
                                    # PLOTTING UTILITIY FUNCTIONS
##############################################################################################################



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
    trans = max_val*((pred_mask > 0.5).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot


##############################################################################################################
                                    # MANUALLY TRACE AN IMAGE USING SIMPLE OPENCV GUI
##############################################################################################################


def manual_trace(img, fname, layer="both", save_path="", save=False):
    '''
    Helper function to manually segment the choroid

    Suggestion from Justin: As more points are added, show the cubic-spline trace.
    '''
    # Sort out directory and filenames
    fname_path = os.path.join(save_path, fname)
    
    rpechor_fn = fname+"_RPEChor"
    rpechor_p = os.path.join(save_path,rpechor_fn)+".csv"
    
    chorscl_fn = fname+"_ChorScl"
    chorscl_p = os.path.join(save_path,chorscl_fn)+".csv"

    # Select pixels along boundary
    if (not os.path.exists(rpechor_p)) or (not os.path.exists(chorscl_p)):

        if os.path.exists(chorscl_p) and layer == "upper":
            lower = pd.read_csv(chorscl_p).values
            if not os.path.exists(rpechor_p):
                select_pixels(img, cmap=None, fname=rpechor_fn, 
                                               save_path=save_path, verbose=0, interp=True)()
            upper = pd.read_csv(rpechor_p).values
        
        elif os.path.exists(rpechor_p) and layer == "lower":
            upper = pd.read_csv(rpechor_p).values
            if not os.path.exists(chorscl_p):
                select_pixels(img, cmap=None, fname=chorscl_fn, 
                                               save_path=save_path, verbose=0, interp=True)()
            lower = pd.read_csv(chorscl_p).values
            
        elif layer == "both":
            if not os.path.exists(rpechor_p):
                select_pixels(img, cmap=None, fname=rpechor_fn, 
                                               save_path=save_path, verbose=0, interp=True)()
            if not os.path.exists(chorscl_p):
                select_pixels(img, cmap=None, fname=chorscl_fn, 
                                               save_path=save_path, verbose=0, interp=True)()
            upper = pd.read_csv(rpechor_p).values
            lower = pd.read_csv(chorscl_p).values
            
    else:
        upper = pd.read_csv(rpechor_p).values
        lower = pd.read_csv(chorscl_p).values

    trace_points = (upper, lower)
    traces = interp_trace(trace_points, align=False)

    # Save out if specified
    if save:
        pickle.dump(traces, open(fname_path+".pkl", "wb"))

    return traces