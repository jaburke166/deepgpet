import numpy as np
import logging
from .pixel_gui import select_pixels

def scale_converter(scale=(11.49,3.87,None)):
    '''
    Given scale, output microns-per-pixel in both horizontal and vertical directions.

    Scale can be defined crudely based on the L-shaped scale shown on the *Heidelberg OCT images, or
    more accurately via the information tab on each OCT image which shows the "Scale X" and "Scale Z" 
    variables.
    '''
    # horizontal-vertical-micron scale constants
    if scale is None:
        scale = (11.49,3.87,None)

    # Scale can also be defined as microns-per-pixel in horizontal and vertical directions, i.e. scale[-1] = None
    pixel_x, pixel_y, micron_scale = scale

    # Microns-per-pixel constants
    if micron_scale is not None:
        micron_pixel_x = micron_scale / pixel_x
        micron_pixel_y = micron_scale / pixel_y
    else:
        micron_pixel_x = pixel_x
        micron_pixel_y = pixel_y

    return micron_pixel_x, micron_pixel_y



def curve_length(curve, scale=(11.49,3.87,None)):
    """
    Calculate the length (in microns) of a curve defined by a numpy array of coordinates.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Scale constants
    xum_per_pix, yum_per_pix = scale_converter(scale)

    # Calculate difference between pairwise consecutive coordinates of curve
    diff = np.abs((curve[1:] - curve[:-1]).astype(np.float64))

    # Convert pixel difference to micron difference
    diff[:, 0] *= xum_per_pix
    diff[:, 1] *= yum_per_pix

    # Length is total euclidean distance between all pairwise-micron-movements
    length = np.sum(np.sqrt(np.sum((diff) ** 2, axis=1)))

    return length


def curve_location(curve, distance=2000, ref_idx=400, scale=(11.49,3.87,None)):
    """
    Given a curve, what two coordinates are *distance* microns away from some coordinate indexed by
    *ref_idx*.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Work out number of microns per unit pixel movement
    N = curve.shape[0]
    
    # Scale constants
    xum_per_pix, yum_per_pix = scale_converter(scale)

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
    if ((idx_l == ref_idx) or (idx_r == ref_idx)) and distance > 200:
        logging.warning(f"Segmentation not long enough for {distance}um distance along trace. Reduce distance value.")
        return None

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


def nearest_coord(trace, coord, offset=15):
    """
    Given a coordinate, find the nearest coordinate along trace and return it.

    INPUTS:
    ------------------------
        trace (np.array) : Upper or lower choroid boundary

        coord (np.array) : Single xy-coordinate.

    RETURNS:
    ------------------------
        trace_refpt (np.array) : Point along trace closest to coord.

        offset_pts (np.array) : Points offset distance from trace_refpt for deducing
            local tangent line.
    """
    N_pts = trace.shape[0]

    # Work out closest coordinate on trace to coord
    fovea_argmin = np.argmin(np.sum((trace - coord) ** 2, axis=1))
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


def _check_perp_x(x_grid, bot_chor):
    '''
    If the lower boundary isn't segmented long enough, the perpendicular
    line drawn fromt the RPE-Chor reference point won't intersect it. This
    function catches this issue.
    '''
    if x_grid.shape[0] == 0:
        x_grid = np.array([bot_chor[-1,0] - bot_chor[0,0]])
        logging.warning("""Lower boundary not segmented far enough to measure exactly perpendicular.
                            \nTaking last coordinate of lower boundary trace as surrogate.""")
    return x_grid


def detect_chorscl_pts(reference_pts, traces, offset=15, plot=True):
    """
    Given the lower choroid boundary and reference points along the upper boundary, detect which
    coordinates along the lower choroid boundary which intersect the perpendicular line drawn from
    a tangent line at these reference points.

    INPUTS:
    ------------------
        reference_pts (np.array) : Points along upper boundary to construct tangent and perpendicular
        lines at. These are where we start measuring CT/CA.

        ttraces (2-tuple) : Tuple storing the upper and lower boundaries of the segmented chorod, in xy-space.

        offset (int) : Value either side of reference point to define tangential line.

        plot (bool) : If flagged, return tangent and perpendicular lines for plotting/visualisation.

    RETURNS:
    ------------------

        chorscl_refpts : Reference points along lower boundary.

        (perpen_lines, tangent_lines) : Perpendicular and tangential lines from reference points.
    """
    # Extract traces
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

    # Evaluate tangent line around the endpoints, rotate 90 degrees and then work out the
    # intersections those perpendicular lines have with the upper and lower choroid
    # boundary, giving us corner-bounds of the area to compute.
    tangent_lines = []
    perpen_lines = []
    chorscl_refpts = np.zeros_like(reference_pts)
    for i, (ref_x, ref_y) in enumerate(reference_pts):

        # Work out local tangent line for each boundary point
        ref_xidx = ref_x - rpechor_stx
        offset_l, offset_r = ref_xidx - offset, ref_xidx + offset
        offset_l, offset_r = _check_offset(offset, (offset_l, offset_r), top_chor.shape[0])
        tan_p1, tan_p2 = top_chor[[offset_l, offset_r]]
        tan_m, tan_c = construct_line(tan_p1, tan_p2)

        # Define tangent line and rotate 90 degrees to get perpendicular line
        n = 150
        if tan_m == np.inf:
            x_pts = np.array(2 * n * [ref_x]).reshape(-1, 1)
            y_pts = np.arange(ref_y - n, ref_y + n).reshape(-1, 1)
        else:
            x_pts = (np.arange(-n, n) + ref_x).reshape(-1, 1)
            y_pts = (x_pts * tan_m + tan_c).reshape(-1, 1)
        tan_line = np.concatenate([x_pts, y_pts], axis=1)

        # Using the reference point as the node on the RPE-Choroid boundary, rotate the tangent line 90 degrees
        perp_y = (tan_line[:, 0] - ref_x + ref_y).reshape(-1, 1)
        perp_x = (-(tan_line[:, 1] - ref_y) + ref_x).reshape(-1, 1)
        perp_line = np.concatenate([perp_x, perp_y], axis=1).astype(int)

        # Append results
        tangent_lines.append(tan_line)
        perpen_lines.append(perp_line)

        # Compute intersections of perpendicular line with upper and lower choroid boundaries
        perp_x_unique = (np.unique(np.rint(perp_x)).astype(int) - chorscl_stx)
        clip_x_unique = (perp_x_unique >= 0) & (perp_x_unique < bot_chor.shape[0])
        perp_x_unique = perp_x_unique[clip_x_unique]
        perp_x_unique = _check_perp_x(perp_x_unique, bot_chor)
        resid = []
        for pixel_coord in bot_chor[perp_x_unique]:
            resid.append(np.sqrt(np.sum((perp_line - pixel_coord) ** 2, axis=1)))
        perp_closest_idx = perp_line[np.argmin(np.vstack(resid)) % perp_line.shape[0]]
        chorscl_refpts[i] = bot_chor[np.argmin(np.sqrt(np.sum((bot_chor - perp_closest_idx) ** 2, axis=1)))]

    if plot:
        outputs = [chorscl_refpts, perpen_lines, tangent_lines]
    else:
        outputs = chorscl_refpts

    return outputs



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