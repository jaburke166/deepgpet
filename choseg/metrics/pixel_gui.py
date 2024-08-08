"""
@author: Jamie Burke
@email s1522100@ed.ac.uk

This script allows a simple GUI to select pixels from an image and save them as part of an image processing pipeline
"""

import os
import pandas as pd
import cv2
import numpy as np
import sys
import skimage
from scipy import interpolate as interp

class select_pixels(object):
    
    def __init__(self, image, scale=800, save_path=sys.path[0], 
                 verbose=1, fname="pixels_selected", 
                 cmap=cv2.COLORMAP_JET, interp=False):
        '''
        This class of functions provides a simple GUI to select pixels from a single image and save them.
        
        INPUTS:
        ---------------------
            image (string or np.array) : Filepath to image or numpy array representing image.
            
            scale (int) : Scale factor to aid showing image on screen.
            
            save_path (str) : Path to directory to save selected pixels.

            verbose (int) : If 1, show coordinate, if 0 only show selected pixel.
            
            fname (str) : Filename to save coordinates to.
            
            cmap (opencv-colormap) : opencv-python colormap. Default as Jet. If None, then Gray is used.

            interp (bool) : If flagged, interpolate currently selected points.
        '''
        # Check if array or filepath
        if isinstance(image, str):
            if os.path.exists(image):
                img = cv2.imread(image, 1)
                image_path = image
            else:
                raise ValueError(f"File path {image} does not exist.")
        else:
            img = image
            image_path = None

        # Convert image to colour by adding third axis
        if img.ndim < 3:
            img = skimage.color.gray2rgb(img)
            
        # Normalise to [0, 255] np.uint8 and apply cmap
        img_norm = self.normalise(img, (0,255), np.uint8)
        if cmap is not None:
            self.img = cv2.applyColorMap(img_norm, cmap)
        else:
            self.img = img_norm
            
        # Define zoom window size, image shape and scaling for showing image
        self.Z = 50
        img_shape = self.img.shape
        self.shape = np.array([img_shape[0], img_shape[1]], dtype=np.float64)
        shape = self.shape / self.shape.max()
        self.ar_size = (int(scale*shape[0]), int(scale*shape[1]))

        # Set up filepaths to intermediary text files storing information on where image was window_zoom at and 
        # which pixels were selected
        self.save_path = save_path
        self.fname = fname
        self.cmap = cmap
        self.verbose=verbose
        self.interp = interp
        self.zoom_txt_path = os.path.join(save_path, 'zoom_xy.txt')
        self.select_pixel_path = os.path.join(save_path, 'selected_pixels.txt')

    def show_img(self, image, image_name='window'):
        '''
        This function shows an image.
        
        INPUTS:
        ----------------------
            image (np.array) : Numpy array representing image.
            
            image_name (string) : If "window", align image window to top-left of screen. If "window_zoom" then
            show just to the right of "window" as this is the zoomed version.
        '''
        # Name the window, resize according to predefined scaling and show image.
        cv2.namedWindow(image_name, flags=cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(image_name, self.ar_size[1], self.ar_size[0])
        cv2.imshow(image_name, image)
            
        # If the window shows the original image then left-align image on screen. Otherwise, display
        # window_zoom image to the right of so it's not occluding the original image
        if image_name == 'window':
            cv2.moveWindow(image_name, 0, 0)
        else:
            cv2.moveWindow(image_name, self.ar_size[1], 0)
            
            
    def normalise(self, img, minmax_val=(0,1), astyp=np.float64):
        '''
        Normalise image between minmax_val.

        INPUTS:
        ----------------
            img (np.array, dtype=?) : Input image of some data type.
            minmax_val (tuple) : Tuple storing minimum and maximum value to normalise image with.
            astyp (data type) : What data type to store normalised image.

        RETURNS:
        ----------------
            img (np.array, dtype=astyp) : Normalised image in minmax_val.
        '''
        # Extract minimum and maximum values
        min_val, max_val = minmax_val

        # Convert to float type to perform [0, 1] normalisation
        img = img.astype(np.float64)

        # Normalise to [0, 1]
        img -= img.min()
        img /= img.max()

        # Rescale to max_val and output as specified data type
        img *= (max_val - min_val)
        img += min_val

        return img.astype(astyp)

            
    def plot_coord(self, image, plot_x, plot_y, image_name='window', window_zoom=False):
        '''
        This function shows the pixel coordinate that the mouse is hovering over on the window
        
        INPUTS:
        ------------------------
            img (np.array) : Numpy array representing image.
            
            plot_x, plot_y (integers) : x- and y-coordinates of pixel.
            
            image_name (string) : The window name of the window to plot coordinate on.
            
            window_zoom (bool) : Flag to tell if hovering over window_zoom version of image or not.
        '''
        # If plotting coordinate on original image, then plot coordinate text "(x,y)" onto image
        if image_name == "window":
            radius = 1
            scale = 1
            offset = 175
            
            # Make sure the text overlaid on the image fits in the window.
            if plot_x < self.shape[1] - offset:
                position = (plot_x, plot_y)
            else:
                position = (plot_x - offset, plot_y)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Remove text if verbose is 0
            if self.verbose:
                cv2.putText(image, '('+str(plot_x)+', '+str(plot_y)+')', 
                            position, 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (245, 245, 245), 2)   
        else:
            radius = 0
       
        # Plot coordinate as a circle of radius, thickness and colour
        cv2.circle(image, (plot_x, plot_y), radius=radius, thickness=-1, color=(245, 245, 245))
    
    
    def show_coords(self, image, zoom=False):
        '''
        This function shows the coordinate each time the mouse moves over the image on the window.
        
        INPUTS:
        -----------------------
            image (np.array) : Numpy array representing image.
                        
            zoom (bool) : Flag for whether coordinate was selected on the original image or the zoomed image.
            If the latter, then we need to add an offset so that the coordinate appears at the right pixel.
        '''
        # Loop over coordinates in selected_pixels.txt
        with open(self.select_pixel_path, "r") as file:
            coords_lst = [np.array(list(map(int, line.strip().split(',')))) for line in file] 
        for (coord_x, coord_y) in coords_lst:
            if zoom:
                plot_x = x+np.max([0, coord_x - self.Z])
                plot_y = y+np.max([0, coord_y - self.Z])
            else:
                plot_x = coord_x
                plot_y = coord_y

            # plot coordinate onto image
            self.plot_coord(image, plot_x, plot_y, "window")

            # Interpolate along selected coordinates
            if self.interp and len(coords_lst)>1:
                coords = np.asarray(coords_lst).reshape(-1,2)
                N_coords = coords.shape[0]
                f = interp.UnivariateSpline(coords[:,0], coords[:,1], k=min(N_coords-1,3))
                x_grid = np.arange(coords[0,0], coords[-1,0])
                f_grid = f(x_grid)
                xf_trace = np.concatenate([x_grid.reshape(-1,1), f_grid.reshape(-1,1)], axis=1).astype(np.int32)
                cv2.polylines(image, [xf_trace], 0, (255,255,255), thickness=3)

            cv2.imshow("window", image)
            

            
    def save_coords(self):
        '''
        Save coordinates found in self.select_pixel_path
        '''
        # Read selected coordinates
        # Loop over coordinates in selected_pixels.txt
        with open(self.select_pixel_path, "r") as file:
            coords_arr = np.array([list(map(int,line.strip().split(','))) for line in file])
        coords_arr = coords_arr[coords_arr[:,0].argsort(axis=0)]
        if self.interp:
            N_coords = coords_arr.shape[0]
            f = interp.UnivariateSpline(coords_arr[:,0], coords_arr[:,1], k=min(N_coords-1,3))
            min_x, max_x = (coords_arr[:,0].min(), coords_arr[:,0].max())
            x_grid = np.arange(min_x, max_x)
            f_grid = f(x_grid)
            coords_arr = np.concatenate([x_grid.reshape(-1,1), f_grid.reshape(-1,1)], axis=1).astype(np.int32)

        coords_df = pd.DataFrame(dict(x=coords_arr[:,0], 
                                      y=coords_arr[:,1]), columns=["x", "y"])
        coords_df.to_csv(os.path.join(self.save_path, f"{self.fname}.csv"), index=False)
            
            
    def click_event_main(self, event, x, y, flags, params):
        '''
        Function to control GUI 
        
        INPUTS:
        ------------------------
            event (object) : This is the window we're working on, could either be the main window, or zoomed window.
            
            x, y (integers) : This is the current pixel the mouse is hovering over
            
            flags (indicators) : These are integers which indicate a particular movement, for example the scroll
            function moving forward or backward.
            
            params (tuple) : Tuple containing an image (could be original or zoomed) with the flag identifying whether
            it is the original image or is the zoomed version.
        '''
        # Load in image and extra parameters on whether image is window_zoom in or not
        img_zoom, window_zoom = params
        img = self.img.copy()
        
        # If window_zoom then change the window name
        if window_zoom:
            image_name = 'window_zoom'
        else:
            image_name = 'window'

        # If the mouse is moving then plot the coordinate onto the original window
        if event == cv2.EVENT_MOUSEMOVE:

            # Plot coordinates on image
            if window_zoom:
                with open(self.zoom_txt_path, "r") as file:
                    zoom_x, zoom_y = np.array(list(map(int, file.readlines()[0].strip().split(","))))
                plot_x = x + max(0, zoom_x - self.Z)
                plot_y = y + max(0, zoom_y - self.Z)
                
                self.plot_coord(img_zoom.copy(), x, y, "window_zoom", window_zoom=True)
            else:
                plot_x = x
                plot_y = y
            
            self.plot_coord(img, plot_x, plot_y, window_zoom=window_zoom)

            # Show current selected pixel coordinates 
            self.show_coords(img)

        # If control button is held as the left-button on the mouse is clicked, save current
        # selection of pictures found in self.select_pixel_path
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            
            # Save selected points into csv file
            self.save_coords()

            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')


        # If right-button on the mouse is clicked, remove all selected pixels to 
        # restart selection process
        if event == cv2.EVENT_RBUTTONDOWN:

            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')


        # If ONLY left-button on the mouse is clicked, show coordinate on original image and
        # zoomed image if window is opened.
        if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):

            # Show current selected pixel coordinates 
            self.show_coords(img)
            
            # Plot coordinates on image
            if window_zoom:
                with open(self.zoom_txt_path, "r") as file:
                    zoom_x, zoom_y = np.array(list(map(int, file.readlines()[0].strip().split(","))))
                plot_x = x + max(0, zoom_x - self.Z)
                plot_y = y + max(0, zoom_y - self.Z)
                
                # Plot coordinates by filling pixel with red on window_zoom image
                self.plot_coord(img_zoom, x, y, 'window_zoom', window_zoom)
            else:
                plot_x = x
                plot_y = y
                
            # Plot coordinate on original image by default
            self.plot_coord(img, plot_x, plot_y, 'window', False)

            # Write pixel coordinate selected to file
            with open(self.select_pixel_path, 'a') as file:
                file.write(f'{plot_x},{plot_y}\n')  
            
        # Given a mouse wheel event and that we've not zoomed in OR our mouse is not on the zoomed in window.
        if event == cv2.EVENT_MOUSEWHEEL and window_zoom == False:

            # Extract size of image
            M, N = int(self.shape[0]), int(self.shape[1])

            # If the mouse wheel is scrolled forward at least once (identified by flags > 0), i.e. zoom-in,
            # open a zoomed version of the original image as a separate window to the right of 
            # the original image window
            if flags > 0:

                # Write to txt file that image was window_zoom and save pixel coordinate selected during
                # mouse scroll
                with open(self.zoom_txt_path, "w") as file:
                    file.write(f'{x},{y}')

                # Extract region of interest of 100 x 100 centred at pixel which zoom occured
                zoom_img = img[max(y-self.Z, 0):min(y+self.Z, M), max(x-self.Z, 0):min(x+self.Z, N)]
                self.zoom_img = zoom_img

                # Show window_zoom image
                self.show_img(zoom_img, 'window_zoom')

                # Setting mouse handler for the image and calling to click_main_event and sending window_zoom img 
                # and window_zoom=True as parameters to this callback function
                cv2.setMouseCallback('window_zoom', self.click_event_main, (zoom_img, True))


            # If the mouse wheel is scrolled backward at least once (identified by flags == 0), i.e. i.e. zoom out
            # reset the zoom by closing zoomed window to just leave original image
            #else:
            #    # Destory window_zoom window
            #    cv2.destroyWindow('window_zoom')

        
    def __call__(self):
        '''
        Call function to run the pixel selection GUI for the inputted image.
        
        INPUTS:
        -------------------
            image (string or np.array): Is either an file path to an image or a numpy array representing
            an image.
        '''
        # show image    
        self.show_img(self.img, image_name='window')
        
        # Input parameters for calling GUI
        params = (self.img, False)

        # Rewrite .txt file to show image_window_zoom=False
        with open(self.zoom_txt_path, 'w') as file:
            file.write('')

        # Remove selected pixels
        with open(self.select_pixel_path, 'w') as file:
            file.write('')

        # Setting mouse handler for the image and calling the click_event() function
        cv2.setMouseCallback('window', self.click_event_main, params)
        
        # wait for Enter/Escape key to be pressed to exit. DO NOT CLOSE WINDOW using "X" button.
        while True:
    
            # it waits till we press a key
            key = cv2.waitKey(0)

            # if we press esc or enter
            if key == 27 or key == 13:
                break
            
        # Once finished processing image delete the txt files
        os.remove(self.zoom_txt_path)
        os.remove(self.select_pixel_path)

        # close the windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)