"""classes for analyzing FRAP data.
Dependencies:
- python >3.5
- czifile
- scipy
- scikit-image
- pandas
- ffmpeg
- matplotlib
- tqdm

Suggested installation: 
Create a new conda environment (optional), install packages with
conda install czifile scipy scikit-image pandas ffmpeg matplotlib tqdm -c conda-forge
in one line.
"""
# Filepaths
import os
from glob import glob

# CZI file API
from czifile.czifile import CziFile

# Arrays, DFs, curve fit
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from lmfit import Model

# Image processing
from scipy.ndimage import gaussian_filter
from skimage.filters.thresholding import threshold_isodata
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.util import view_as_windows

# Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.gridspec as grd
from matplotlib.patches import Circle

# Progress bars, warnings
from tqdm import tqdm

# String parsing
import re

# Type hints and func signatures
from typing import Tuple, Dict
from inspect import signature

# Pickle
import pickle


class FRAPChannel:
    """Object representing a channel of a FRAP movie from a CZI file. 
    Attributes are parsed from the CZI metadata and cached for analysis.
    Bleaching must occur once and the bleach spot must be a circle.

    init:
    ----
    filepath            :   str, path to CZI file
    channel             :   int, channel number of CZI file to analyze

    attributes:
    ----------
    filepath            :   str, valid path to CZI file
    channel             :   int, channel number
    Czi                 :   czifile.Czi object associated with the file
    metadata            :   dict, convenience attribute for metadata
    channel_name        :   str, channel name extracted from metadata
    movie               :   np.ndarray, movie data for this channel only
    timestamps          :   np.ndarray, timestamps relative to the
                            bleach stop time (i.e., negative before
                            bleach event), in seconds.
    n_frames            :   int, number of frames in this movie
    roi_size            :   Tuple[int, int], ROI size in (Y, X)
    bleach_circ_center  :   Tuple[int, int], center of the bleach
                            spot in (Y, X). Extracted from metadata.
    bleach_diameter_px  :   float, diameter of the circular bleach spot,
                            extracted from metadata.
    bleach_frame        :   int, frame after bleaching event, zero-
                            indexed. For example, if "Wait 20 frames
                            before bleaching" was selected, this is 20.
    elapsed_time_sec    :   float, last timestamp of experiment minus
                            first timestamp of experiment in seconds
    frame_intervals_sec :   np.ndarray, time gaps between frames after
                            the bleach frame.
    """
    def __init__(self, filepath: str, channel: int=0):
        assert os.path.isfile(filepath) and \
               os.path.splitext(filepath)[-1] == '.czi', \
                f"{filepath} is not a valid CZI file!"
        
        self.filepath = filepath
        self.channel = channel
        self.Czi = CziFile(self.filepath)
        self.metadata = self.Czi.metadata(raw=False)\
            ['ImageDocument']['Metadata']
        self.movie = self.preprocess_movie()
        
    def __repr__(self):
        """String representation of FRAPChannel."""
        return "FRAPChannel object:\n    {}".format("\n    ".join([
            f"{attr : <20}:    {getattr(self, attr)}" 
            for attr in ['filepath', 'channel_name', 'n_frames', 
                         'roi_size', 'bleach_frame', 'bleach_diameter_px', 
                         'bleach_circ_center', 'elapsed_time_sec']]))

    def preprocess_movie(self):
        # Get movie in a specific axis order
        dims = ['T', 'Y', 'X', 'C', 'Z']
        orig_indices = [self.Czi.axes.index(d) for d in dims]
        movie = np.moveaxis(self.Czi.asarray(), orig_indices, [0, 1, 2, 3, 4])
        
        # If movie has z-slices, apply a sum-intensity projection
        if movie.shape[dims.index('Z')] > 1:
            movie = np.sum(movie, axis=dims.index('Z'))
        # Check if channel passed is valid
        try:
            movie = movie[:, :, :, self.channel, ...]
        except IndexError as error:
            print(f"CZI movie has {movie.shape[dims.index('C')]} channel(s),",
                  f"channel index {self.channel} is not valid for this movie")
            raise error
        movie = np.squeeze(movie)   # Squeeze out 1-length dimensions
        assert movie.ndim == 3, "FRAPChannel.preprocess_movie: \
            Movie has incorrect number of dimensions."

        # Crop movie if acquisition box doesn't agree with frame size
        try:
            self.geo = self.metadata['Layers']['Layer']['Elements']\
                                    ['Rectangle']['Geometry']
            sl = np.s_[:,round(self.geo['Top']):
                        round(self.geo['Top']+self.geo['Height']),
                        round(self.geo['Left']):
                        round(self.geo['Left']+self.geo['Width'])]
            return movie[sl]
        
        except:
            implied_X = self.metadata['Information']['Image']['SizeX']
            implied_Y = self.metadata['Information']['Image']['SizeY']
            if (implied_X != movie.shape[2]) or (implied_Y != movie.shape[1]):
                raise RuntimeWarning("FRAPChannel.preprocess_movie: \
                    Warning: movie size doesn't match metadata.")
            return movie

    ################
    ## PROPERTIES ##
    ################
    @property
    def channel_name(self) -> str:
        """Get channel name from metadata"""
        if not hasattr(self, "_channel_name"):
            self.parse_metadata()
        return self._channel_name

    @property
    def timestamps(self) -> np.ndarray:
        """numpy.ndarray of shape (n_frames,) representing timestamps
        relative to the bleach stop timestamp."""
        if not hasattr(self, "_timestamps"):
            self.parse_attachments()
        return self._timestamps
    
    @property
    def n_frames(self) -> int:
        """Number of frames in channel."""
        if not hasattr(self, "_n_frames"):
            self._n_frames = self.movie.shape[0]
        return self._n_frames
    
    @property
    def roi_size(self) -> Tuple[int, int]:
        """Size of ROI imaged in pixels."""
        if not hasattr(self, "_roi_size"):
            self._roi_size = (self.movie.shape[1], self.movie.shape[2])
        return self._roi_size
    
    @property
    def bleach_circ_center(self) -> Tuple[float, float]:
        if not hasattr(self, "_bleach_circ_center"):
            self.parse_metadata()
        return self._bleach_circ_center
       
    @property
    def bleach_diameter_px(self) -> float:
        """Diameter of the circular bleach spot in pixels."""
        if not hasattr(self, "_bleach_diameter_px"):
            self.parse_metadata()
        return self._bleach_diameter_px

    @property
    def bleach_frame(self) -> int:
        """Returns frame immediately after bleach. 
        Zero-indexed, i.e. if "wait 20 frames before 
        bleaching" was used to image, this will be 20."""
        if not hasattr(self, "_bleach_frame"):
            self.parse_metadata()
        return self._bleach_frame
    
    @property
    def elapsed_time_sec(self) -> float:
        """Returns elapsed time in seconds."""
        if not hasattr(self, "_elapsed_time_sec"):
            self._elapsed_time_sec = self.timestamps.max() \
                                   - self.timestamps.min()
        return self._elapsed_time_sec
    
    @property
    def frame_intervals_sec(self) -> np.ndarray:
        """Time gaps between frames after the bleach frame. 
        Has shape (n_frames-1,)."""
        if not hasattr(self, "_frame_intervals_sec"):
            t = self.timestamps[self.bleach_frame:]    # after bleach
            self._frame_intervals_sec = t[1:] - t[:-1] # differences
        return self._frame_intervals_sec

    #############
    ## METHODS ##
    #############
    def parse_attachments(self):
        """Parse CZI file to get timestamps relative to bleach."""
        timestamps = self.Czi.attachment_directory[1].data_segment().data()
        events = self.Czi.attachment_directory[0].data_segment().data()
        # Find bleach stop timestamp
        if isinstance(events, Tuple):
            id = [e.__str__().startswith("BLEACH_STOP") 
                  for e in events].index(True)
        else:
            raise RuntimeError("FRAPChannel.parse_attachments:\
                Failed to parse attachments on CZI file.")
        
        bleach_stop = float(re.findall(r'\b\d[\d,.]*\b', 
                            events[id].__str__())[0])
        self._timestamps = timestamps - bleach_stop
    
    def parse_metadata(self):
        """Parse metadata to get bleach_frame, bleach spot diameter, 
        bleach spot center, and channel_name."""
        timeline = self.metadata['Information']\
            ['TimelineTracks']['TimelineTrack']
        # Figure out where the bleach info is encoded
        if isinstance(timeline, dict) and timeline['Name'] == 'Bleaching Track':
            bleach_track = timeline['TimelineElements']['TimelineElement']
        elif isinstance(timeline, list):
            id = [entry['Name'] for entry in timeline].index('Bleaching Track')
            bleach_track = timeline[id]['TimelineElements']['TimelineElement']
        else:
            raise RuntimeError("FRAPChannel.parse_metadata: Failed to parse\
                metadata. Are there multiple bleach events in your CZI movie?")
        self._bleach_frame = bleach_track['Bounds']['StartT']
        
        # Parse SVG string to get circle center and diameter
        svg = bleach_track['EventInformation']['Bleaching']\
            ['BleachingRegions']['BleachRegion']['BleachRegionGeometryString']
        points = re.findall("\d+\.\d+", svg)
        assert len(points) % 2 == 0, \
            f"FRAPChannel.parse_metadata: Failed to parse bleach geometry."
        points = [(float(points[i]), float(points[i+1])) 
                  for i in range(0, len(points), 2)]
        points = np.asarray(points)
        # Return as y, x Tuple
        center = (np.median(points[:,1]), np.median(points[:,0]))
        self._bleach_circ_center = (center[0] - self.geo['Top'],
                                    center[1] - self.geo['Left'])
        #self._bleach_circ_center = (np.median(points[:,1]), 
        #                            np.median(points[:,0]))
        
        # Take the average of the x range and y range as the bleach diameter
        x_range = np.max(points[:,0]) - np.min(points[:,0])
        y_range = np.max(points[:,1]) - np.min(points[:,1])
        self._bleach_diameter_px = np.average([x_range, y_range])
        
        channels = self.metadata['DisplaySetting']['Channels']['Channel']
        # Figure out where channel name is encoded
        if isinstance(channels, dict):
            self._channel_name = channels['Name']
        elif isinstance(channels, list):
            self._channel_name = channels[self.channel]['Name']
        else:
            raise RuntimeError("FRAPChannel.parse_metadata: Failed to \
                                parse metadata for channel name.")

    ##############
    ## PLOTTING ##
    ##############
    def plot_bleach_center(self, show_plot: bool=True, out_png: str=None):
        """Plot the center of the bleach spot, taken from the CZI metadata, 
        on top of the pre-bleach and bleach frames. Mostly a sanity check."""
        _, axs = plt.subplots(1, 3)
        axs[0].imshow(self.movie[self.bleach_frame-1])   # Pre-bleaching
        axs[1].imshow(self.movie[self.bleach_frame])     # Frame after bleach
        axs[2].imshow(self.movie[self.bleach_frame])

        # Show center
        y, x = self.bleach_circ_center
        axs[2].scatter(x, y, s=3.0, color='r')

        # Turn off all axes
        for ax in axs:
            ax.axis('off')
        
        # Save if desired
        if out_png is not None:
            plt.savefig(out_png, dpi=800)
        
        if show_plot:
            plt.show()
        plt.close()
    
    def plot_frap_montage(self, 
                          frames: list, 
                          shape: Tuple[int, int], 
                          labels: list=None, 
                          show_plot: bool=True, 
                          out_path: str=None):
        """Plot a montage of by showing the frames in *frames*, 
        optionally with a list of *labels*."""
        assert len(frames) == shape[0] * shape[1], \
            "Shape and number of frames don't match"
        if labels is not None:
            assert len(labels) == shape[0] * shape[1], \
                "Shape and number of labels must match"

        _, axs = plt.subplots(*shape)
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(self.movie[frames[i]])
            ax.axis('off')
            if labels is not None:
                ax.set_title(labels[i], fontsize='small')
        
        axs[0].add_artist(AnchoredSizeBar(axs[0].transData,
                                        5/(20/259), 
                                        "5 $\mu$m", 
                                        "lower left",
                                        bbox_to_anchor=(0, 0),
                                        bbox_transform=axs[0].transAxes,
                                        pad=0.1,
                                        color='white',
                                        frameon=False))
        
        if out_path is not None:
            plt.savefig(out_path, dpi=800, 
                        transparent=True, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        plt.close()


class FRAPAnalysis:
    """Class for analyzing a FRAPChannel. Caches analysis parameters
    as well as calculated values like masks and intensities.
    
    init:
    -----
    channel             :   FRAPChannel object
    window_width        :   int, width of the sliding window used to
                            track the FRAP spot throughout the movie
    max_shift           :   int, maximum number of pixels the 
                            FRAP spot can move in one frame
    mask_gaussian_sigma :   float, sigma of the gaussian filter
                            used to make a binary nuclear mask
    background_subtract :   bool, if True, the FRAPChannel.movie is 
                            modified by median-subtracting non-nuclear 
                            pixels from each frame of the movie
    
    attributes:
    -----------
    filepath            :   str, FRAPChannel.filepath
    bleach_frame        :   int, FRAPChannel.bleach_frame
    timestamps          :   np.ndarray, FRAPChannel.timestamps
    nuclear_masks       :   np.ndarray, binary nuclear masks for
                            every frame of FRAPChannel.movie
    movie               :   np.ndarray, median-background-subtracted
                            if background_subtract is True, else
                            identical to FRAPChannel.movie
    init_spot_center    :   Tuple[int, int], initial center of the
                            bleach spot, rounded to the nearest int
    spot_centers        :   np.ndarray, centers of the bleach 
                            spot in each frame
    spot_masks          :   np.ndarray, binary masks for the bleach
                            spot in each frame
    nuclear_intensities :   np.ndarray, sum intensity of the 
                            nuclear mask in each frame
    spot_intensities    :   np.ndarray, sum intensity of the
                            spot mask in each frame
    norm_intensities    :   np.ndarray, spot intensity divided by
                            nuclear intensity in each frame, normalized
                            to the pre-bleach average intensity.
    bleach_depth        :   float, 1 - norm intensity after bleach
    norm_intensity_df   :   pd.DataFrame of timestamps 
                            and norm_intensities          
    """
    def __init__(self, channel: FRAPChannel, window_width: int=81, 
            max_shift: int=1, mask_gaussian_sigma: float=3.0,
            background_subtract: bool=True, mask_mode='by_frame'):
        self.channel = channel
        self.window_width = window_width
        self.max_shift = max_shift
        self.mask_gaussian_sigma = mask_gaussian_sigma
        self.background_subtract = background_subtract
        self.mask_mode = 'by_frame' if mask_mode != 'by_movie' else 'by_movie'

    def __repr__(self):
        """String representation of FRAPAnalysis."""
        return "FRAPAnalysis object:\n    {}".format("\n    ".join([
            f"{attr : <20}:    {getattr(self, attr)}" 
            for attr in ['filepath', 'window_width', 'max_shift', 
                         'mask_gaussian_sigma', 'background_subtract', 
                         'mask_mode']]))

    ################
    ## PROPERTIES ##
    ################
    @property
    def filepath(self) -> str:
        """Filepath of FRAPChannel analyzed in this object"""
        return self.channel.filepath
        
    @property
    def bleach_frame(self) -> int:
        """Frame immediately after bleach event, zero-indexed"""
        return self.channel.bleach_frame
    
    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps relative to bleach stop, in seconds"""
        return self.channel.timestamps

    @property
    def nuclear_masks(self) -> np.ndarray:
        """Make nuclear masks by thresholding a Gaussian-blurred 
        image of each frame in the FRAPChannel.movie."""
        if not hasattr(self, "_nuclear_masks"):
            masks = np.zeros(self.channel.movie.shape, dtype=int)
            if self.mask_mode == 'by_frame':
                for i in range(self.channel.movie.shape[0]):
                    blur = gaussian_filter(self.channel.movie[i], 
                        sigma=self.mask_gaussian_sigma).astype('int64')
                    mask = blur > threshold_isodata(blur)
                    masks[i] = remove_small_holes(
                        remove_small_objects(mask, 2000), 5000)
            elif self.mask_mode == 'by_movie':
                max_int_proj = self.normalize_image(self.channel.movie[0])
                for frame in self.channel.movie[1:]:
                    max_int_proj = np.maximum(max_int_proj, 
                                              self.normalize_image(frame))
                blur = gaussian_filter(max_int_proj, 
                    sigma=self.mask_gaussian_sigma).astype('int64')
                mask = blur > threshold_isodata(blur)
                mask = remove_small_holes(
                    remove_small_objects(mask, 2000), 5000)
                masks[:,] = mask
            self._nuclear_masks = masks
        return self._nuclear_masks

    @property
    def movie(self) -> np.ndarray:
        """np.ndarray corresponding to the movie data in the 
        FRAPChannel object. This movie is median-subtracted 
        if background_subtract is True, otherwise it is unmodified."""
        if not hasattr(self, "_movie"):
            if self.background_subtract:
                background = np.logical_not(self.nuclear_masks).astype(float)
                background[background == 0] = np.nan
                background = background * self.channel.movie.astype(float)
                medians = np.nanmedian(background, axis=(1,2))
                self._movie = (self.channel.movie.T - medians).T
            else:
                self._movie = self.channel.movie
        return self._movie

    @property
    def init_spot_center(self) -> Tuple[int, int]:
        """Center of the initial bleach spot, in (Y, X) and rounded"""
        return(int(np.around(self.channel.bleach_circ_center[0])), 
               int(np.around(self.channel.bleach_circ_center[1])))
    
    @property
    def spot_centers(self) -> np.ndarray:
        """Center of the bleach spot as it drifts throughout the movie,
        calculated by the maximum of the image correlation between the
        spot immediately after bleaching and each subsequent frame."""
        if not hasattr(self, "_spot_centers"):
            self.calculate_spot_center_drift()
        return self._spot_centers

    @property
    def spot_masks(self) -> np.ndarray:
        """Circular masks drawn around every spot center in every frame"""
        if not hasattr(self, "_spot_masks"):
            masks = np.zeros(self.movie.shape)
            for frame in range(self.movie.shape[0]):
                y, x = np.ogrid[:self.movie.shape[1], :self.movie.shape[2]]
                dist_from_center = np.sqrt(
                    (y - self.spot_centers[0, frame])**2 \
                  + (x - self.spot_centers[1, frame])**2)
                masks[frame] = dist_from_center \
                            <= (self.channel.bleach_diameter_px/2)
            self._spot_masks = masks
        return self._spot_masks
    
    @property
    def nuclear_intensities(self) -> np.ndarray:
        """Sum intensity of the nuclear mask in each frame"""
        if not hasattr(self, "_nuclear_intensities"):
            self._nuclear_intensities = np.sum(self.nuclear_masks * self.movie, 
                                               axis=(1,2))
        return self._nuclear_intensities
    
    @property
    def spot_intensities(self) -> np.ndarray:
        """Sum intensity of the spot mask in each frame"""
        if not hasattr(self, "_spot_intensities"):
            self._spot_intensities = np.sum(self.spot_masks * self.movie,
                                            axis=(1,2))
        return self._spot_intensities
    
    @property
    def norm_intensities(self) -> np.ndarray:
        """Spot intensity divided by nuclear intensity in each frame,
        normalized to the average intensity of the frames pre-bleach."""
        if not hasattr(self, "_norm_intensities"):
            norm_int = self.spot_intensities / self.nuclear_intensities
            prebleach_int = np.average(norm_int[:self.bleach_frame])
            self._norm_intensities = norm_int / prebleach_int
        return self._norm_intensities
    
    @property
    def bleach_depth(self) -> float:
        """1 - normalized intensity after bleach frame"""
        if not hasattr(self, "_bleach_depth"):
            self._bleach_depth = 1 - self.norm_intensities[self.bleach_frame]
        return self._bleach_depth
    
    @property
    def proportion_recovered(self) -> float:
        """The proportion of 1-bleach_depth was recovered in the movie."""
        if not hasattr(self, "_proportion_recovered"):
            intensities = self.norm_intensities[self.timestamps > 0]
            max_intensity = np.min([np.max(intensities), 1.0])
            self._proportion_recovered = (max_intensity - (1-self.bleach_depth)) \
                                       / self.bleach_depth
        return self._proportion_recovered
    
    @property
    def time_half_recovery(self) -> float:
        """Time in seconds until half-recovery"""
        if not hasattr(self, "_time_half_recovery"):
            half_recovery = (1 - self.bleach_depth) + (self.bleach_depth / 2)
            idx = np.where((self.norm_intensities > half_recovery)[self.bleach_frame:])[0] \
                          + self.bleach_frame
            if idx.size == 0:
                self._time_half_recovery = np.nan
            elif idx.size > 0:
                self._time_half_recovery = self.timestamps[np.min(idx)]
        return self._time_half_recovery
    
    @property
    def intensity_df(self) -> pd.DataFrame:
        """pd.DataFrame with timestamps and normalized 
        intensities, useful for saving to CSV."""
        if not hasattr(self, "_intensity_df"):
            self._intensity_df = pd.DataFrame(dict(
                timestamp=self.timestamps,
                nuc_intensity=self.nuclear_intensities,
                spot_intensity=self.spot_intensities,
                normalized_intensity=self.norm_intensities))
        return self._intensity_df

    #############
    ## METHODS ##
    #############
    def calculate_spot_center_drift(self) -> np.ndarray:
        """Get FRAP spot centers as it drifts throughout the movie."""
        def crop_image(image: np.ndarray, 
                       center: Tuple[int, int], 
                       pad: int) -> np.ndarray:
            """Crop an image around a center pixel by 
            adding pad around it on all four sides"""
            return image[center[0]-pad:center[0]+pad+1,
                         center[1]-pad:center[1]+pad+1]
        
        def xcorr2_min(image: np.ndarray, template: np.ndarray) -> np.ndarray:
            """2D local cross-correlation of template across windows of image.
            Output shape is (w_image-w_template+1, h_image-h_template+1)"""
            assert len(image.shape) == len(template.shape) == 2, \
                "At least one of the inputs is not a 2D image!"
            assert image.shape >= template.shape, \
                f"Image ({image.shape}) is larger than template \
                ({template.shape}); args may be swapped."
            
            # Sliding windows of shape template convolved with template
            out_arr = np.square(view_as_windows(image, template.shape) 
                              - template)
            out_arr = np.sum(out_arr, axis=(2,3))   # Sum of each convolution
            arr_min = np.asarray(np.unravel_index(np.argmin(out_arr), 
                                                  out_arr.shape))

            # Return offset by subtracting index of max 
            # value relative to the center of the array
            arr_center = np.asarray([out_arr.shape[0] // 2, 
                                     out_arr.shape[1] // 2])
            return arr_min - arr_center
        
        init_center = self.init_spot_center
        centers = np.zeros((2, self.movie.shape[0]), dtype=int)
        pad = self.window_width // 2
        prebleach = crop_image(self.movie[self.bleach_frame-1], 
                               init_center, 
                               pad)
        bleach  = crop_image(self.movie[self.bleach_frame], 
                             init_center, 
                             pad)
        template = self.normalize_image(prebleach + bleach)

        y = init_center[0]
        x = init_center[1]
        for fr in range(self.bleach_frame, self.movie.shape[0]-1):
            try:
                image = self.normalize_image(crop_image(self.movie[fr+1], 
                                                        (y, x), 
                                                        pad+self.max_shift))
                y_off, x_off = xcorr2_min(image, template)
            except:
                print(f"error in file {self.channel.filepath}")
                return
            y += y_off
            x += x_off
            centers[0, fr+1] = y
            centers[1, fr+1] = x
        centers[0, :self.bleach_frame+1] = centers[0, self.bleach_frame+1]
        centers[1, :self.bleach_frame+1] = centers[1, self.bleach_frame+1]

        self._spot_centers = centers
    
    ##############
    ## PLOTTING ##
    ##############
    def animate_masks_normalized_intensity(self, 
                                           show_plot: bool=True, 
                                           out_mp4: str=None):
        """Animate the CZI movie, the spot mask, and the 
        nuclear mask. Also plot the normalized intensity."""
        if (not show_plot) and (out_mp4 is None):
            print("Plot will not be shown or saved, skipping")
            return

        # Figure and axes setup
        fig = plt.figure()    
        gs = grd.GridSpec(2, 4, figure=fig, hspace=0.1)
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        axs.append(fig.add_subplot(gs[1, :]))

        # Plot first frame of movies and save as variable to change later
        im = axs[0].imshow(self.movie[0], animated=True)
        nuc = axs[1].imshow(self.nuclear_masks[0], cmap='gray', animated=True)
        spot = axs[2].imshow(self.spot_masks[0], cmap='gray', animated=True)
        overlay = axs[3].imshow(self.movie[0], animated=True)
        circ = axs[3].add_patch(Circle(
            (self.spot_centers[1,0], self.spot_centers[0,0]), 
            self.channel.bleach_diameter_px/2,
            color='w', fill=False, animated=True))

        # Initialize axis bounds with all intensity data, then show first point
        line, = axs[4].plot(self.timestamps, self.norm_intensities)  
        line.set_data(self.timestamps[0], self.norm_intensities[0])      
        
        # Turn axis markings off for movie plots
        for ax in axs[:4]:
            ax.axis('off')
        
        axs[4].set_xlabel("Time relative to bleach (s)")
        axs[4].set_ylabel("Normalized intensity")
        
        # Animation function to get subsequent frames
        def animate(step):
            im.set_data(self.movie[step])
            nuc.set_data(self.nuclear_masks[step])
            spot.set_data(self.spot_masks[step])
            overlay.set_data(self.movie[step])
            circ.center = self.spot_centers[1,step], self.spot_centers[0,step]
            line.set_data(self.timestamps[:step+1], 
                          self.norm_intensities[:step+1])
            return im, nuc, spot, overlay, line, circ

        ani = animation.FuncAnimation(fig, animate, frames=self.movie.shape[0], 
                                      interval=40,  # 40 ms frame interval
                                      blit=True)    # Only draw changed px
        if show_plot:
            plt.show()
        plt.close()

        if out_mp4 is not None:
            print(f"Saving animation to {out_mp4}, this could take a minute...")
            writer = animation.FFMpegWriter(fps=15, bitrate=1800)
            ani.save(out_mp4, writer=writer)
    
    def plot_spot_and_nuclear_intensity(self,
                                        normalize=True, 
                                        show_plot: bool=True, 
                                        out_png: str=None):
        """Plot the spot intensity and nuclear mask 
        intensity over time on parasite axes."""
        if (not show_plot) and (out_png is None):
            print("Plot will not be shown or saved, skipping")
            return
            
        if normalize:
            nuc_intensities = self.nuclear_intensities \
                            / np.mean(self.nuclear_intensities[:self.bleach_frame])
            spot_intensities = self.spot_intensities \
                             / np.mean(self.spot_intensities[:self.bleach_frame])
            ylabel = "Normalized intensity"
        else:
            nuc_intensities = self.nuclear_intensities \
                            / np.sum(self.nuclear_masks, axis=(1,2))
            spot_intensities = self.spot_intensities \
                             / np.sum(self.spot_masks, axis=(1,2))
            ylabel = "Average intensity"

        plt.plot(self.timestamps, nuc_intensities, label="nuclear mask")
        plt.plot(self.timestamps, spot_intensities, label="spot mask")
        plt.ylabel(ylabel)
        plt.xlabel("Time relative to bleach (sec)")
        plt.legend()

        # Save if desired
        if out_png is not None:
            plt.savefig(out_png, dpi=800)

        # Show plot, if desired
        if show_plot:
            plt.show()
        plt.close()

    ###########
    ## UTILS ##
    ###########
    @staticmethod    
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize an image by mean-subtracting and 
        dividing by its standard deviation."""
        return (image-image.mean()) / np.std(image)
    

class FRAPCondition:
    """Class encapsulating data corresponding to one condition."""
    def __init__(self, paths: Dict[str, int], condition_name: str=None, 
                 progress_bar:bool=True, **kwargs):
        # Check that list is not empty
        assert bool(paths), "FRAPCondition: no paths passed"
        
        # Check all filepaths passed by trying to make a FRAPChannel
        for k, v in paths.items():
            _ = FRAPChannel(k, channel=v)
        
        # Pass on valid analysis kwargs
        allowed_kwargs = ["window_width", "max_shift", "mask_gaussian_sigma", 
                          "background_subtract", "mask_mode"]
        self.kwargs = dict([(k, v) for k, v in kwargs.items() 
                            if k in allowed_kwargs])

        self.paths = paths
        self.n_files = len(paths)
        self.progress_bar = progress_bar
        self.condition_name = condition_name \
            if condition_name is not None else "unnamed_condition"


    def __repr__(self):
        """String representation of FRAPDataset."""
        return "FRAPCondition object:\n    {}".format("\n    ".join([
            f"{attr : <20}:    {getattr(self, attr)}" 
            for attr in ['condition_name', 'n_files']]))
    
    @classmethod
    def from_constant_channel(cls, 
                              paths: list, 
                              condition_name: str=None, 
                              channel: int=0, 
                              progress_bar: bool=True, 
                              **kwargs):
        """Convenience constructor. Most users will want to analyze
        the same channel across multiple movies as one condition."""
        if isinstance(paths, str):
            paths = [paths]
        
        return cls(dict([(p, channel) for p in paths]), condition_name, 
                   progress_bar, **kwargs)

    ################
    ## PROPERTIES ##
    ################
    @property
    def timestamps(self) -> np.ndarray:
        if not hasattr(self, "_timestamps"):
            self.get_timestamps_norm_intensities()
        return self._timestamps
    
    @property
    def norm_intensities(self) -> np.ndarray:
        if not hasattr(self, "_norm_intensities"):
            self.get_timestamps_norm_intensities()
        return self._norm_intensities
    
    @property
    def pos_timestamps(self) -> np.ndarray:
        """Sorted array of positive timestamps only, 
        corresponding to timestamps after bleaching."""
        if not hasattr(self, "_pos_timestamps"):
            self.get_timestamps_norm_intensities()
        return self._pos_timestamps
    
    @property
    def pos_intensities(self) -> np.ndarray:
        """Array of normalized intensities corresponding to 
        positive timestamps only, sorted by positive timestamps."""
        if not hasattr(self, "_pos_intensities"):
            self.get_timestamps_norm_intensities()
        return self._pos_intensities
    
    @property
    def bleach_depths(self) -> np.ndarray:
        if not hasattr(self, "_bleach_depths"):
            self.get_timestamps_norm_intensities()
        return self._bleach_depths
    
    @property
    def prop_recovered(self) -> np.ndarray:
        if not hasattr(self, "_prop_recovered"):
            self.get_timestamps_norm_intensities()
        return self._prop_recovered
    
    @property
    def time_half_recovered(self) -> np.ndarray:
        if not hasattr(self, "_time_half_recovered"):
            self.get_timestamps_norm_intensities()
        return self._time_half_recovered
    
    @property
    def frame_intervals_sec(self) -> np.ndarray:
        """Get all the time differences between subsequent frames in seconds"""
        if not hasattr(self, "_frame_intervals_sec"):
            intervals = np.array([], dtype=float)
            for path, channel in self.paths.items():
                a = self._init_FRAPAnalysis(path, channel)
                intervals = np.append(intervals, a.channel.frame_intervals_sec)
            self._frame_intervals_sec = intervals
        return self._frame_intervals_sec
    
    @property
    def rolling_mean_intensities(self) -> Tuple:
        """length-3 tuple of ndarrays of shape (n,), where n is number 
        of intensities in this FRAPCondition. The first axis is the 
        rolling timestamp means, the second is the intensity means, 
        and the third is the stdev at each point."""
        if not hasattr(self, '_rolling_mean_intensities'):
            rolling_timestamps = sliding_window_view(self.pos_timestamps, 
                                                     self.n_files*2)
            rolling_intensities = sliding_window_view(self.pos_intensities, 
                                                      self.n_files*2)
            mean_times = np.mean(rolling_timestamps, axis=1)
            mean_intensities = np.mean(rolling_intensities, axis=1)
            std = np.std(rolling_intensities, axis=1)

            self._rolling_mean_intensities = (mean_times,
                                              mean_intensities,
                                              std)
        return self._rolling_mean_intensities
        
    @property 
    def single_exp_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns an lmfit ModelResult object 
        for a single exponential fit."""
        if not hasattr(self, "_single_exp_fit"):
            # Fit a single exponential using lmfit. Bounds for A 
            # are (0, 1) and bounds for tau are (0, infinity).
            single_exp_model = Model(self._single_exp)
            single_exp_model.set_param_hint('A', value=0.8, min=0, max=1)
            single_exp_model.set_param_hint('tau', value=1, min=0)
            single_exp_model.set_param_hint('C', value=0.2, 
                                            min=0, max=1, expr='1 - A')     
            fit_result = single_exp_model.fit(self.pos_intensities,
                                              dt=self.pos_timestamps)
            if fit_result.success:
                self._single_exp_fit = fit_result
            else:
                print(f"Single exponential fit failed for {self.condition_name}")

        return self._single_exp_fit
    
    @property
    def double_exp_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns an lmfit ModelResult object
        for a double exponential fit."""
        if not hasattr(self, "_double_exp_fit"):
            # Fit a double exponential using lmfit. Bounds for A terms 
            # are (0, 1) and bounds for tau terms are (0, infinity).
            double_exp_model = Model(self._double_exp)
            double_exp_model.set_param_hint('A1', value=0.4, min=0, max=1)
            double_exp_model.set_param_hint('tau1', value=1, min=0)
            double_exp_model.set_param_hint('A2', value=0.4, min=0, max=1)
            double_exp_model.set_param_hint('tau2', value=10, min=0)
            double_exp_model.set_param_hint('C', value=0.2, 
                                            min=0, max=1, expr='1 - A1 - A2')
            fit_result = double_exp_model.fit(self.pos_intensities,
                                              dt=self.pos_timestamps)
            if fit_result.success:
                self._double_exp_fit = fit_result
            else:
                print(f"Double exponential fit failed for {self.condition_name}")
        
        return self._double_exp_fit

    #############
    ## METHODS ##
    #############    
    def get_timestamps_norm_intensities(self):
        """Get timestamps and normalized intensities as flattened ndarrays"""
        timestamps = np.array([], dtype=float)
        norm_intensities = np.array([], dtype=float)
        bleach_depths = np.zeros((self.n_files))
        prop_recovered = np.zeros((self.n_files))
        time_half_recovered = np.zeros((self.n_files))

        # Set up progress bar
        if self.progress_bar:
            iterations = enumerate(tqdm(self.paths.items(),
                                   "Calculating normalized intensities..."))
        else:
            iterations = self.paths.items()

        for i, (path, channel) in iterations:
            a = self._init_FRAPAnalysis(path, channel)
            timestamps = np.append(timestamps, a.timestamps)
            norm_intensities = np.append(norm_intensities, a.norm_intensities)
            bleach_depths[i] = a.bleach_depth
            prop_recovered[i] = a.proportion_recovered
            time_half_recovered[i] = a.time_half_recovery
        
        assert timestamps.shape == norm_intensities.shape, \
            "Error in FRAPCondition.get_timestamps_norm_intensities: \
            timestamps and normalized intensities are different lengths."
        self._timestamps = timestamps
        self._norm_intensities = norm_intensities
        self._bleach_depths = bleach_depths
        self._prop_recovered = prop_recovered
        self._time_half_recovered = time_half_recovered

        # Cache positive timestamps only and their associated 
        # intensities, which are used for fitting
        pos_timestamps = timestamps[timestamps > 0]
        pos_intensities = norm_intensities[timestamps > 0]
        self._pos_timestamps = np.sort(pos_timestamps)
        self._pos_intensities = pos_intensities[np.argsort(pos_timestamps)]
    
    def print_fits(self):
        """Pretty-print single- and double-exponential fits"""
        print(f"Single exponential fit (R-squared = {self.single_exp_fit.summary()['rsquared']}):",
              self.single_exp_fit.fit_report(show_correl=False).split("[[Variables]]")[-1])
        print(f"Double exponential fit (R-squared = {self.double_exp_fit.summary()['rsquared']}):",
              self.double_exp_fit.fit_report(show_correl=False).split("[[Variables]]")[-1])
    
    def print_stats(self):
        """Pretty-print some non-parametric statistics."""
        for attr in ['bleach_depths', 'prop_recovered', 'time_half_recovered']:
            print(f"{attr : <20}:\
            {np.nanmean(getattr(self, attr)):.4f} ± {np.nanstd(getattr(self, attr)):.4f}")
    
    def to_pickle(self, out_pkl: str):
        """Pickle FRAPCondition"""
        with open(out_pkl, 'wb') as pkl:
            pickle.dump(self, pkl)

    ##############
    ## PLOTTING ##
    ##############
    def plot_scatterplot(self,
                         plot_rolling_mean: bool=True,  
                         plot_rolling_std: bool=False,
                         plot_single_exp: bool=False, 
                         plot_double_exp: bool=True,
                         plot_predicted_error: dict={},
                         show_plot: bool=True,
                         ylim: Tuple[float, float]=None, 
                         out_png: str=None):
        """Plot normalized intensity versus time for all movies.
        args:
        -----
        plot_rolling_mean   :   bool, plot the rolling average 
                                intensity as a line
        plot_rolling_std    :   bool, plot the rolling standard deviation
                                as a shaded region of the plot.
        plot_single_exp     :   bool, plot the single-exponential fit
                                as a blue line.
        plot_double_exp     :   bool, plot the double-exponential fit
                                as a red line.
        plot_predicted_error:   dict, plot the uncertaintly of the fit
                                as a shaded region of the plot.
                                Keys can be either "single_exp" or
                                "double_exp", and values should be
                                integers corresponding to the sigma
                                of the fit uncertainty. 
                                Ex.: {"double_exp": 1} will plot the
                                standard deviation of the double-
                                exponential fit as a shaded region.
        show_plot           :   bool, show the plot
        ylim                :   Tuple[float, float], y-axis limits
        out_png             :   str, path to save the plot as a PNG

        output:
        -------
        show                :   a plot of normalized intensity versus 
                                time if show_plot is True
        save                :   PNG to out_png if out_png is not None
        
        """
        # Plot normalized intensities
        plt.scatter(x=self.timestamps, y=self.norm_intensities, 
                    s=2, color='dimgray')        
        
        # Plot the rolling average intensities and std, if desired
        mean_times, mean_intensities, std = self.rolling_mean_intensities
        if plot_rolling_mean:
            plt.plot(mean_times, mean_intensities, color='k')

        if plot_rolling_std:
            plt.fill_between(x=mean_times, 
                             y1=mean_intensities+std,
                             y2=mean_intensities-std,
                             alpha=0.3, 
                             color='gray')

        # Plot exponential fits, if desired
        if plot_single_exp:
            plt.plot(self.pos_timestamps, 
                     self.single_exp_fit.best_fit, 
                     'b-', 
                     linewidth=2)

        if plot_double_exp:
            plt.plot(self.pos_timestamps, 
                     self.double_exp_fit.best_fit, 
                     'r-', 
                     linewidth=2)
        
        # Plot fit error, if desired
        if plot_predicted_error:
            if "single_exp" not in plot_predicted_error and \
               "double_exp" not in plot_predicted_error:
                print("plot_predicted_error must be a dict with keys \
                      'single_exp' and/or 'double_exp'. No errors plotted.")
            if "single_exp" in plot_predicted_error:
                try:
                    _ = self.single_exp_fit.eval_uncertainty(sigma=plot_predicted_error["single_exp"])
                    del_y_pred = self.single_exp_fit.dely_predicted
                    plt.fill_between(x=self.pos_timestamps, 
                                     y1=self.single_exp_fit.best_fit + del_y_pred,
                                     y2=self.single_exp_fit.best_fit - del_y_pred,
                                     alpha=0.3, 
                                     color='b')
                except:
                    print("Values for dictionary 'plot_predicted_error' must be integers.")
            if "double_exp" in plot_predicted_error:
                try:
                    _ = self.double_exp_fit.eval_uncertainty(sigma=plot_predicted_error["double_exp"])
                    del_y_pred = self.double_exp_fit.dely_predicted
                    plt.fill_between(x=self.pos_timestamps, 
                                     y1=self.double_exp_fit.best_fit + del_y_pred,
                                     y2=self.double_exp_fit.best_fit - del_y_pred,
                                     alpha=0.3, 
                                     color='r')
                except:
                    print("Values for dictionary 'plot_predicted_error' must be integers.")

        # Axis labels
        plt.xlabel("Time relative to bleach (s)")
        plt.ylabel("Normalized intensity")

        # y-axis limits
        if ylim is not None:
            plt.ylim(ylim)
        
        if out_png is not None:
            plt.savefig(out_png, dpi=800)
        
        if show_plot:
            plt.show()
        plt.close()
    
    def plot_residuals(self, show_plot: bool=True, out_png: str=None):
        """Plot the residuals of both the single-exponential
        and double-exponential fit."""
        _, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
        self.single_exp_fit.plot_residuals(ax=axs[0], datafmt='k.', data_kws={'alpha': 0.5})
        self.double_exp_fit.plot_residuals(ax=axs[1], datafmt='k.', data_kws={'alpha': 0.5})
        axs[0].set_title("Single-exponential residuals")
        axs[1].set_title("Double-exponential residuals")
        axs[1].set_xlabel("Time relative to bleach (s)")

        if out_png is not None:
            plt.savefig(out_png, dpi=800)
        
        if show_plot:
            plt.show()
        plt.close()
    
    def plot_frame_interval_histogram(self, 
                                      show_plot: bool=True, 
                                      out_png: str=None):
        """Plot a histogram of frame intervals of the movies in this
        FRAPCondition. Some microscopes just can't keep time."""
        plt.hist(self.frame_intervals_sec, bins=30)
        plt.xlabel("Frame interval (s)")
        plt.ylabel("Count")
        plt.title("Histogram of frame intervals")

        if out_png is not None:
            plt.savefig(out_png, dpi=800)
        
        if show_plot:
            plt.show()
        plt.close()
    
    def animate_all_masks_intensities(self, 
                                      show_plots: bool=False, 
                                      save_mp4s: bool=True,
                                      out_folder: str=None):
        """Animate all mask/intensity plots for files in paths.
        By default, saves animations to the save folder where CZI
        movies were if an out_folder is not specified."""
        if not show_plots and not save_mp4s:
            print(f"Animations will not be shown or saved, skipping.")
            return
        
        # Figure out what paths to save to
        for path, channel in self.paths.items():
            a = self._init_FRAPAnalysis(path, channel)
            if save_mp4s and out_folder is not None:
                out_name = os.path.basename(a.filepath)
                out_name = os.path.splitext(out_name)[0]
                out_name = os.path.join(out_folder, 
                                        f"{out_name}_c{a.channel.channel}.mp4")    
            elif save_mp4s and out_folder is None:
                out_name = os.path.splitext(a.filepath)[0]
                out_name = f"{out_name}_c{a.channel.channel}.mp4"
            else:
                out_name = None
            
            a.animate_masks_normalized_intensity(show_plot=show_plots,
                                                 out_mp4=out_name)
    
    def plot_all_spot_nuclear_intensities(self,
                                          normalize=True, 
                                          show_plots: bool=False, 
                                          save_pngs: bool=True,
                                          out_folder: str=None):
        """Plot all nuclear and spot intensity plots for files in paths.
        By default, saves animations to the save folder where CZI
        movies were if an out_folder is not specified."""
        if not show_plots and not save_pngs:
            print(f"Plots will not be shown or saved, skipping.")
            return
        
        # Figure out what paths to save to
        for path, channel in self.paths.items():
            a = self._init_FRAPAnalysis(path, channel)
            if save_pngs and out_folder is not None:
                out_name = os.path.basename(a.filepath)
                out_name = os.path.splitext(out_name)[0]
                out_name = os.path.join(out_folder, 
                                        f"{out_name}_c{a.channel.channel}.png")    
            elif save_pngs and out_folder is None:
                out_name = os.path.splitext(a.filepath)[0]
                out_name = f"{out_name}_c{a.channel.channel}.png"
            else:
                out_name = None
            
            a.plot_spot_and_nuclear_intensity(show_plot=show_plots,
                                              normalize=normalize, 
                                              out_png=out_name)
                    
    ###########
    ## UTILS ##
    ###########
    def _init_FRAPAnalysis(self, filepath: str, channel: int) -> FRAPAnalysis:
        """Make a FRAPAnalysis"""
        return FRAPAnalysis(FRAPChannel(filepath, channel), **self.kwargs)
    
    @staticmethod
    def _single_exp(dt, A, tau):
        C = 1 - A
        return A * (1 - np.exp(-dt / tau)) + C

    @staticmethod
    def _double_exp(dt, A1, tau1, A2, tau2):
        C = 1 - A1 - A2
        return (A1 * (1 - np.exp(-dt / tau1))
              + A2 * (1 - np.exp(-dt / tau2))
              + C)


class FRAPDataset:
    """NOT FUllY IMPLEMENTED"""
    def __init__(self, condition_dict: Dict[str, list[FRAPAnalysis]]):
        # Sanity checks for passed dictionary
        for k, v in condition_dict.items():
            # Put FRAPAnalysis object in list if it's the 
            # only one in a condition and not in a list
            if not isinstance(v, list):
                v = [v]
                condition_dict[k] = v

            # Check for empty lists
            assert bool(v), f"Condition {k} has an empty list"

            # Check all elements in list are FRAPAnalysis
            for i in v:
                assert isinstance(i, FRAPAnalysis), \
                    f"An item in condition {k} is not a FRAPAnalysis object."
        
        self.condition_dict = condition_dict
        self.conditions = list(condition_dict.keys())
        self.n_files = [len(v) for v in condition_dict.values()]

    def __repr__(self):
        """String representation of FRAPDataset."""
        return "FRAPDataset object:\n    {}".format("\n    ".join([
            f"{attr : <20}:    {getattr(self, attr)}" 
            for attr in ['conditions', 'n_files']]))
    
    ################
    ## PROPERTIES ##
    ################
    @property
    def timestamps(self) -> np.ndarray:
        if not hasattr(self, '_timestamp_dict'):
            d = self.condition_dict.copy()
            for k in d.keys():
                arr = np.array([], dtype=float)
                for v in d[k]:
                    pass

    ##############
    ## PLOTTING ##
    ##############  

if __name__ == "__main__":
    folder = '/Volumes/VF4/FRAP/221104_LnCAPFRAP'

    key = 'H2B'
    files = glob(os.path.join(folder, f"*{key}*.czi"))
    cond = FRAPCondition.from_constant_channel(files, condition_name=key, mask_mode='by_movie')
    #cond.animate_all_masks_intensities()
    cond.plot_scatterplot(out_png=os.path.join(folder, f"{key}_scatterplot.png"))
    #cond.plot_all_spot_nuclear_intensities()
    cond.to_pickle(os.path.join(folder, f"{key}.pkl"))
    cond.print_stats
    cond.print_fits

