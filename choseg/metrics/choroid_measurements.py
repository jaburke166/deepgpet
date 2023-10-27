import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import logging
import sys
from pathlib import Path
from .chor_utils import (curve_length, curve_location, scale_converter, 
                    nearest_coord, construct_line, detect_chorscl_pts)

sys.path.append(Path().absolute().parent)
from choseg.utils import (extract_bounds, interp_trace)

def compute_choroid_thickness_area(mask, 
                                   fovea: [tuple, np.ndarray] = None,
                                   scale: tuple[int, int, int] = (11.49,3.87,None),
                                   macula_rum: int = 3000,
                                   N_measures: int = 3,
                                   N_avgs = 0,
                                   offset=15,
                                   return_lines: bool = False,
                                   return_distances: bool = False):
    """
    Given a binary mask, extract boundaries truncated to ROI of label mask. If fovea is visible,
    use that to reference where to measure CA/CT from. Otherwise, use the middle point of the label.
    Measure CA 3000 microns either side of this reference point. Measure CT at 3 points,
    one underneath the reference point, and two points measured 1500/3000 microns each side of reference point.

    INPUTS:
    ----------------
        mask (np.array) : Binary mask.

        fovea (np.array) : xy-space coordinate of the fovea. If None, then assumed to be evaluating either a volume
            image (without a fovea) or it is simply unknown.

        scale (3-tuple) : x-y-micron scale dictated by the device.

        macula_rum (int) : Micron distance from fovea to measure CT and CA from. Must be odd.

        N_measures (int) : Number of choroid thickness measurements to make. Must be odd.

        N_avgs (int) : Number of point measures to average over for CT measurement. If 0, then a single point measure (which is what
        is typically used in the literature). Must be even, to have an even number either side of actual reference point.

        return_lines (bool) : If flagged, return the coordinates along both boundaries to measure thickness with.

        return_distances (bool) : If flagged, return distances from fovea at which measurements were made.

    RETURNS:
    -----------------
        choroid_thickness (np.array) : Integer array storing thicknesses at each location.

        choroid_area (float) : Float measured as square millimetres
    """
    # Scale constants
    N_measures = N_measures + 1 if N_measures % 2 == 0 else N_measures
    micron_pixel_x, micron_pixel_y = scale_converter(scale)
    pixels_per_micron  = 1 / micron_pixel_x     

    # Traces
    if isinstance(mask, np.ndarray):
        traces = interp_trace(extract_bounds(mask), align=False)

    # If a tuple, then is already list of trace boundaries
    elif isinstance(mask, tuple):
        traces = interp_trace(mask)

    # For computing tangent either side of reference point along top boundary
    # offset = 15, currently user defined

    # If fovea is known
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea

    # If not, take column as halfway along trace
    else:
        x_N = int(0.5 * (traces[0].shape[0] + traces[1].shape[0]))
        # x_st = int(0.5*(traces[0,0] + traces[1,0]))
        x_st = int(0.5 * (traces[0][0, 0] + traces[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = traces[0][:,1].mean()
        ref_pt = np.array([x_half, y_half])

    # Extract traces and re-adjust reference point on upper boundary to align with
    # indexing
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = int(top_chor[0, 0]), int(bot_chor[0, 0])
    rpechor_refpt, offset_pts = nearest_coord(top_chor, ref_pt, offset)
    ref_idx = rpechor_refpt[0] - rpechor_stx

    # Set up list of distances either side of reference point, dictated by N_measures
    delta_micron = 2 * macula_rum // (N_measures - 1)
    delta_pixel = int(delta_micron * pixels_per_micron)
    delta_i = [i for i in range((N_measures - 1) // 2 + 1)]
    delta_fulli = [i for i in range(-(N_measures - 1) // 2, (N_measures - 1) // 2 + 1)]
    micron_measures = np.array([i * delta_micron for i in delta_i])
    if return_distances:
        CT_dist_lbls = [delta_micron*delta_fulli[i] for i in range(N_measures)]

    # Locate coordinates along the upper boundary at equal spaces away from foveal pit until macula_rum
    curve_indexes = [curve_location(top_chor, distance=d,
                                    ref_idx=ref_idx, scale=scale) for d in micron_measures]

    # To catch if we cannot make measurement macula_rum either side of reference point.
    for idx in curve_indexes:
        if idx is None:
            logging.warning("Could not make measurement macula_rum either side of reference point. Returning 0s.")
            return np.zeros(N_measures), 0

    # Collect reference points along upper boundary
    N_avgs = N_avgs+1 if N_avgs%2 != 0 else N_avgs
    rpechor_pts = np.unique(np.array([top_chor[[idx + np.arange(-N_avgs//2, N_avgs//2+1)]] for loc in curve_indexes for idx in loc]).reshape(-1,2), axis=0)
    chorscl_pts = detect_chorscl_pts(rpechor_pts, traces, offset, plot=False).reshape(N_measures, N_avgs+1, 2)

    # Compute reference points along Choroid-Scleral boundary, given the reference points along the RPE-Choroid boundary
    boundary_pts = np.concatenate([rpechor_pts.reshape(*chorscl_pts.shape), chorscl_pts], axis=-1).reshape(*chorscl_pts.shape, 2)
                                       
    # Compute choroid thickness at each boundary point. If averaging 
    delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)) * np.array([micron_pixel_x, micron_pixel_y])
    choroid_thickness = np.rint(np.sqrt(np.square(delta_xy).sum(axis=-1)).mean(axis=1)).astype(int)

    # Compute choroid area                               
    area_bnds_arr = np.swapaxes(boundary_pts[[0,-1], N_avgs//2], 0, 1).reshape(-1,2)
    choroid_area = _compute_area_enclosed(traces, area_bnds_arr, scale=scale, plot=return_lines)

    if return_lines*return_distances == 1:
        return choroid_thickness, choroid_area, boundary_pts, CT_dist_lbls
    elif return_lines:
        return choroid_thickness, choroid_area, boundary_pts
    elif return_distances:
        return choroid_thickness, choroid_area, CT_dist_lbls
    else:
        return choroid_thickness, choroid_area


def _compute_area_enclosed(traces, 
                           area_bnds_arr, 
                           scale=(11.49,3.87,None), 
                           plot=False):
    """
    Function which, given traces and four vertex points defining the smallest irregular quadrilateral to which
    the choroid area is enclosed in, calculate the area to square millimetres.

    INPUTS:
    ---------------------
        traces (3-tuple) : Tuple storing upper and lower boundaries of trace

        area_bnds_arr (np.array) : Four vertex pixel coordnates defining the smallest irregular quadrilateral
            which contains the choroid area of interest.

        scale (3-tuple) : Tuple storing pixel_x-pixel_y-micron scalar constants.

        plot (bool) : If flagged, output information to visualise area calculation, including the points contained
            in the quadrilateral and the smallest rectangular which contains the irregular quadrilateral.

    RETURNS:
    --------------------
        choroid_mm_area (float) : Choroid area in square millimetres.

        plot_outputs (list) : Information to plot choroid area calculation.
    """
    # Extract reference points scale and traces
    top_chor, bot_chor = traces#interp_trace(traces, align=False)
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]
    rpechor_ref, chorscl_ref = area_bnds_arr[:2], area_bnds_arr[2:]

    # Compute microns-per-pixel and how much micron area a single 1x1 pixel represents.
    micron_pixel_x, micron_pixel_y = scale_converter(scale)
    micron_area = micron_pixel_x * micron_pixel_y
    
    # Work out range of x- and y-coordinates bound by the area, building the smallest rectangular
    # region which overlaps the area of interest fully
    x_range = np.arange(area_bnds_arr[:, 0].min(), area_bnds_arr[:, 0].max() + 1)
    y_range = np.arange(min(top_chor[x_range[0] - rpechor_stx: x_range[-1] - rpechor_stx + 1, 1].min(),
                            area_bnds_arr[:, 1].min()),
                        max(bot_chor[x_range[0] - chorscl_stx: x_range[-1] - chorscl_stx + 1, 1].max(),
                            area_bnds_arr[:, 1].max()) + 1)
    N_y = y_range.shape[0]

    # This defines the left-most perpendicular line and right-most perpendicular line
    # for comparing with coordinates from rectangular region
    left_m, left_c = construct_line(rpechor_ref[0], chorscl_ref[0])
    right_m, right_c = construct_line(rpechor_ref[1], chorscl_ref[1])
    if left_m != np.inf:
        left_x = ((y_range - left_c) / left_m).astype(np.int64)
    else:
        left_x = np.array(N_y * [rpechor_ref[0][0]])
    if right_m != np.inf:
        right_x = ((y_range - right_c) / right_m).astype(np.int64)
    else:
        right_x = np.array(N_y * [rpechor_ref[1][0]])
    # The rectangular region above needs reduced to only containing coordinates which lie
    # above the Chor-Sclera boundary, below the RPE-Choroid boundary, to the right of the
    # left-most perpendicular line and to the left of the right-most perpendicular line.
    keep_pixel = []

    # We vectorise check by looping across x_range and figuring out if each coordinate
    # in the column satisfies the four checks described above
    for x in x_range:
        # Extract column
        col = np.concatenate([x * np.ones(N_y)[:, np.newaxis], y_range[:, np.newaxis]], axis=1)

        # Define upper and lower bounds at this x-position
        top, bot = top_chor[x - rpechor_stx], bot_chor[x - chorscl_stx]

        # Check all 4 conditions and make sure they are all satisfied
        cond_t = col[:, 1] >= top[1]
        cond_b = col[:, 1] <= bot[1]
        cond_l = x >= left_x
        cond_r = x < right_x
        col_keep = col[cond_t & cond_b & cond_l & cond_r]
        keep_pixel.append(col_keep)

    # All pixels bound within the area of interest
    keep_pixel = np.concatenate(keep_pixel)

    # Calculate area (in square mm)
    choroid_pixel_area = keep_pixel.shape[0]
    choroid_mm_area = np.round(1e-6 * micron_area * choroid_pixel_area, 6)

    # If plotting, reutrn pixels used to compute  area
    if plot:
        plot_outputs = [keep_pixel, (x_range, y_range), (left_x, right_x)]
        outputs = [choroid_mm_area, plot_outputs]
    else:
        outputs = choroid_mm_area

    return outputs


def compute_choroid_crosssectionarea(mask, 
                                     fovea=None, 
                                     macula_rum=3000, 
                                     scale=(11.49,3.87,None), 
                                     return_lines=False):
    '''
    Given the location of the fovea (regardless of it being superior/inferior to the fovea) and a radius
    from the fovea to measure the area to, compute area approximately parallel to choroid (dictated by local
    tangent at fovea along RPE-Choroid trace).
    '''
    # Traces
    if isinstance(mask, np.ndarray):
        traces = interp_trace(extract_bounds(mask), align=False)

    # If a tuple, then is already list of trace boundaries
    elif isinstance(mask, tuple):
        traces = interp_trace(mask)
    
    # Extract traces and scales
    offset = 15
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

    # If macula_rum is None, then compute total available area, dictated by the ends of the traces
    # This is used for computing RNFL area
    if (macula_rum is None) and (fovea is None):
        rpechor_st_en, chorscl_st_en = top_chor[[0,-1]], bot_chor[[0,-1]]
        area_bnds_arr = np.concatenate([rpechor_st_en, chorscl_st_en], axis=0)

    # if macula_rum and fovea are specified, then area is determined on macula scans
    else:
        # locate nearest coordinat on RPE-C boundary to selected foveal coordinate
        # and construct tangent line to this cooardinate
        rpechor_refpt, offset_pts = nearest_coord(top_chor, fovea, offset=offset)
        ref_idx = rpechor_refpt[0] - rpechor_stx
        curve_indexes = curve_location(top_chor, distance=macula_rum, ref_idx=ref_idx, scale=scale)
        if curve_indexes is None:
            return None
        reference_pts = np.asarray([top_chor[idx] for idx in curve_indexes])

        # Compute reference points along Choroid-Scleral boundary, given the reference points
        # along the RPE-Choroid boundary
        chorscl_refpts = detect_chorscl_pts(reference_pts, traces, offset=offset, plot=False)
        area_bnds_arr = np.concatenate([reference_pts, chorscl_refpts], axis=0)

    # Compute choroid area
    choroid_mm_area, plot_info = _compute_area_enclosed(traces, area_bnds_arr, scale, plot=True)

    if return_lines:
        additional_output = [area_bnds_arr, plot_info]
        return choroid_mm_area, *additional_output
    else:
        return choroid_mm_area


def compute_choroid_volume(masks, 
                           fovea: [tuple, np.ndarray] = None,
                           scale: tuple[int, int, int] = (11.49,3.87,None),
                           macula_rum: int = 3000,
                           slice_num: int = 15):
    """
    Given a stack of masks for a given macula Ppole scan, compute the choroid volume and the subregional volumes.
    
    NOTE: Because of a lack of sample size for each quadrant, the sum of subregional volumes do not equal the total
    volume. This needs addressed and worked on.
    
    INPUTS:
    ----------------
        mask (np.array) : Array of binary masks. Can input traces instead

        fovea (np.array) : xy-space coordinate of the fovea. If None, then assumed to be evaluating either a volume
            image (without a fovea) or it is simply unknown.
            
        scale (3-tuple) : x-y-micron scale dictated by the device.

        macula_rum (int) : Radius of macular volume to compute in microns


    RETURNS:
    -----------------
        choroid_volume (np.array) : Float measured as cubic millimetres.

        choroid_subr (float) : List of volume measured for the four quadrants (superior-inferior-nasal-temporal).
    """
    # Metric constants. delta_zum is distance between consecutive B-scans 
    # (240 microns for a 31-slice volume stack, 120 microns for 61-slice, etc.)
    N_scans = len(masks)
    fovea_slice_N = N_scans // 2 if slice_num is None else slice_num 
    delta_zum = 120 if N_scans == 61 else 240

    # If masks is an array, we assume stack of binary masks
    if isinstance(masks, np.ndarray):
        traces = []
        for i in range(N_scans):
            traces.append(extract_bounds(masks[i]))

    # If a list, then is already list of trace boundaries
    elif isinstance(masks, list):
        traces = masks.copy()
    else:
        logging.warning(f"Input not astype np.array or list.")
        return 0

    # If fovea is known
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    # If not, take middle column of ground truth choroid mask.
    else:
        fovea_trace = traces[fovea_slice_N]
        fovea_mask = masks[fovea_slice_N]
        x_N = int(0.5 * (fovea_trace[0].shape[0] + fovea_trace[1].shape[0]))
        x_st = int(0.5 * (fovea_trace[0][0, 0] + fovea_trace[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = fovea_mask.shape[0] // 2
        ref_pt = np.array([x_half, y_half])

    # Compute cross-sectional areas across macular scans
    chor_areas = []
    chor_dict_areas = dict()
    dum_from_fovea = []
    all_traced = True
    for i in range(-macula_rum // delta_zum + 1, macula_rum // delta_zum + 1):

        s_i = fovea_slice_N + i
        dum_from_fovea.append(i * delta_zum)
        macula_i_rum = np.sqrt(macula_rum ** 2 - (i * delta_zum) ** 2)
        s_trace = traces[s_i]

        # Try computing area
        try:
            choroid_area = compute_choroid_crosssectionarea(traces=s_trace, fovea_coord=ref_pt,
                                                            macula_rum=macula_i_rum, scale=scale)
        except:
            logging.warning(f"Could not make measurement {macula_rum} either side of reference point. Returning 0.")
            return 0

        if choroid_area is None:
            logging.warning(f"Could not make measurement {macula_i_rum} either side of reference point for image {s_i}. Returning 0.")
            return 0
            
        chor_areas.append(choroid_area)
        chor_dict_areas[s_i] = choroid_area
    chor_mm2_areas = np.asarray(chor_areas)

    # Approximate volume
    # Micron distance across macula between slices, coverted to mm
    dmm_from_fovea = 1e-3 * np.asarray(dum_from_fovea)
    slice_z = 1e-3 * delta_zum
    chor_mm3_volume = integrate.simpson(chor_mm2_areas, dx=slice_z, even='avg', axis=-1)

    return chor_mm3_volume, chor_dict_areas


def compute_choroid_volume_subregions(masks, 
                                      fovea: [tuple, np.ndarray] = None,
                                      scale: tuple[int, int, int] = (11.49,3.87,None),
                                      macula_rum: int = 3000,
                                      eye: str = "OD",
                                      slice_num: int = 15,):
    """
    Given a stack of masks for a given macula Ppole scan, compute the choroid volume and the subregional volumes.
    
    NOTE: Because of a lack of sample size for each quadrant, the sum of subregional volumes do not equal the total
    volume. This needs addressed and worked on.
    
    INPUTS:
    ----------------
        mask (np.array) : Array of binary masks. Can input traces instead

        fovea (np.array) : xy-space coordinate of the fovea. If None, then assumed to be evaluating either a volume
            image (without a fovea) or it is simply unknown.

        scale (3-tuple) : x-y-micron scale dictated by the device.

        macula_rum (int) : Radius from fovea to define macular region.

        eye (str) : Type of eye (so as to swap temporal/nasal labelling). Should be automated in future.

    RETURNS:
    -----------------
        choroid_volume (np.array) : Float measured as cubic millimetres.

        choroid_subr (float) : List of volume measured for the four quadrants (superior-inferior-nasal-temporal).
    """
    # Region labelling depending on if volume is for right eye or left eye
    if eye == "OD":
        regions = ["Superior", "Nasal", "Inferior", "Temporal"] 
    elif eye == "OS":
        regions = ["Superior", "Temporal", "Inferior", "Nasal"]

    # Metric constants. delta_zum is distance between consecutive B-scans (240 microns for a 
    # 31-slice volume stack, 120 microns for 61-slice, etc.)
    N_scans = len(masks)
    fovea_slice_N = N_scans // 2 if slice_num is None else slice_num 
    delta_zum = 120 if N_scans == 61 else 240
    offset=15

    # Offset the radial distance if we don't have enough scans to sample choroid area to reduce measurement error
    # between choroid volume and sum of subregional volumes
    if N_scans == 31:
        if macula_rum > 500:
            macula_rum += 20
        else:
            macula_rum += 100

    # If masks is an array, we assume stack of binary masks
    if isinstance(masks, np.ndarray):
        traces = []
        for i in range(N_scans):
            traces.append(extract_bounds(masks[i]))

    # If a list, then is already list of trace boundaries
    elif isinstance(masks, list):
        traces = masks.copy()
    else:
        logging.warning(f"Input not astype np.array or list.")
        return 0

    # If fovea is known
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    # If not, take middle column of ground truth choroid mask.
    else:
        fovea_trace = traces[fovea_slice_N]
        fovea_mask = masks[fovea_slice_N]
        x_N = int(0.5 * (fovea_trace[0].shape[0] + fovea_trace[1].shape[0]))
        x_st = int(0.5 * (fovea_trace[0][0, 0] + fovea_trace[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = fovea_mask.shape[0] // 2
        ref_pt = np.array([x_half, y_half])

    # We loop over all slices which are contained within the volume specified by macula_rum
    subr_areas = {region:[] for region in regions}
    subr_dict_areas = {region:{} for region in regions}
    for i in range(-macula_rum // delta_zum+1, macula_rum // delta_zum+1):

        # Work out index in scan Z-stack we are processing and how many microns to measures either side
        # of fovea. Extract trace
        s_i = fovea_slice_N + i
        macula_i_rum = np.sqrt(macula_rum ** 2 - (i * delta_zum) ** 2)
        s_trace = traces[s_i]
        top_chor, bot_chor = s_trace
        rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]

        # locate nearest coordinat on RPE-C boundary to selected foveal coordinate
        # and construct tangent line to this cooardinate
        rpechor_refpt, offset_pts = nearest_coord(top_chor, ref_pt, offset=offset)
        ref_idx = rpechor_refpt[0] - rpechor_stx
        curve_indexes = curve_location(top_chor, distance=macula_i_rum, ref_idx=ref_idx, scale=scale)
        if curve_indexes is None:
            logging.warning(f"Could not make measurement {macula_rum} either side of reference point. Returning 0.")
            return 0

        # For non-foveal slice, work out indexes along RPE-C boundary where we split the area into their
        # respective subregions. For foveal slice, we only have 2 subregions, for area to contribute to 
        # temporal and nasal regions.
        if s_i != fovea_slice_N:

            # Work out distance in microns from fovea which defines where two subregions of the 
            # macular volume meet
            subr_dum = np.abs(i) * (delta_zum)

            # If the chord line distance (macula_i_rum) is greater than where the temporal/nasal reference
            # lines intersect the chord line itself, we must split area into central, nasal and temporal
            # subregions. Otherwise, area only contributes centrally (i.e. to superior/inferior)
            if macula_i_rum >= subr_dum: 
                curve_sub_idxs = curve_location(top_chor, distance=subr_dum, ref_idx=ref_idx, scale=scale)
                rpe_subR = np.asarray([top_chor[idx] for idx in [curve_indexes[0], curve_sub_idxs[0]]])
                rpe_subC = np.asarray([top_chor[idx] for idx in curve_sub_idxs])
                rpe_subL = np.asarray([top_chor[idx] for idx in [curve_sub_idxs[1], curve_indexes[1]]])
                rpe_pts = [rpe_subR, rpe_subC, rpe_subL]

                # Defines the subregion index to add areas to. 
                # 3 - Temporal, 2 - Inferior, 1 - Nasal, 0 - Superior
                subregion_idxs = [3, 2, 1] if i < 0 else [3, 0, 1]
            else:
                rpe_subC = np.asarray([top_chor[idx] for idx in curve_indexes])
                rpe_pts =[rpe_subC]
                subregion_idxs = [2] if i < 0 else [0]
  
        else:
            
            rpe_subR = np.asarray([top_chor[idx] for idx in [curve_indexes[0], ref_idx]])
            rpe_subL = np.asarray([top_chor[idx] for idx in [ref_idx, curve_indexes[1]]])
            rpe_pts = [rpe_subR, rpe_subL]
            subregion_idxs = [3, 1]

        # Detect the points along the choroid-scleral boundary where we define the smallest irregular
        # quadrilateral around the enclosed choroid area.
        chor_pts = []
        for i, rpe_ref in enumerate(rpe_pts):
            chor_pts.append(detect_chorscl_pts(rpe_ref, s_trace, offset=offset, plot=False))

        # Calculate the choroid area given an enclosed area using rpe_pts and chor_pts and add to one 
        # of the subregion lists
        for j, (rpe_ref, chor_ref) in enumerate(zip(rpe_pts, chor_pts)):
            area_bnds_arr = np.concatenate([rpe_ref, chor_ref], axis=0)
            area = _compute_area_enclosed(s_trace, area_bnds_arr, scale, plot=False)
            subr_areas[regions[subregion_idxs[j]]].append(area) #[s_i] = area
            subr_dict_areas[regions[subregion_idxs[j]]][s_i] = area

    # With subregional areas, we can go ahead and compute volumes 
    slice_z = 1e-3 * delta_zum
    subr_volumes = {region:[] for region in regions}
    for key in subr_areas.keys():
       subr_volumes[key] = integrate.simpson(subr_areas[key], dx=slice_z, even='avg', axis=-1)

    return subr_volumes, subr_dict_areas



def compute_etdrs_volume(subr_vols, 
                         subr_lims=(500,1500,3000), 
                         eye="OD"):
    '''
    Given subregional volumes, compute choroid volume across the standardised ETDRS study

    Input is assumed to be a dictionary which at the first level corresponds to
    the radial distance from fovea, and second level corresponds to the regional
    locations across the macula
    '''
    low, mid, high = subr_lims
    if eye == "OD":
        etdrs_locs = ["Superior", "Nasal", "Inferior", "Temporal"]
    elif eye == "OS":
        etdrs_locs = ["Superior", "Nasal", "Inferior", "Temporal"]
    
    choroid_etdr_vol = {}
    etdrs_grid = {low:"central", mid:"inner", high:"outer"}
    etdrs_subrgrid = ["central"] + [" ".join([grid, loc]) for grid in list(etdrs_grid.values())[1:] for loc in etdrs_locs]
    for loc in etdrs_subrgrid:
        if loc == "central":
            choroid_etdr_vol[loc] = sum(list(subr_vols[low].values()))
        else:
            in_out, reg_loc = loc.split(" ")
            
            if in_out=="inner":
                choroid_etdr_vol[loc] = subr_vols[mid][reg_loc] - subr_vols[low][reg_loc]
            elif in_out == "outer":
                choroid_etdr_vol[loc] = subr_vols[high][reg_loc] - subr_vols[mid][reg_loc]

    return choroid_etdr_vol