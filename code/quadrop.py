# File management
import glob
import os
import shutil
import csv

# Data processing
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from skimage import io
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from PIL import Image, ImageEnhance, ImageOps

# Utilities
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
mp.set_start_method('fork', force=True)
from ipywidgets import interact, FloatSlider, Layout, interactive
import random
from tqdm import tqdm
import itertools
import cv2
from natsort import natsorted
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from scipy.ndimage import gaussian_filter1d  # Import for Gaussian smoothing

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



"""
Name: 
    Quantitative Active Drops Phenotyping (QuADroP)
Title:
    quadrop.py
Last update:
    2023-11-09
Author(s):
    David Larios
Purpose:
    This file compiles all of the relevant functions for processing raw
    PIV data from Matlab PIVlab for the ActiveDROPS project.
"""

######################################### style #########################################
# Default RP plotting style
def set_plotting_style():
    """
    Formats plotting environment to that used in Physical Biology of the Cell,
    2nd edition. To format all plots within a script, simply execute
    `mwc_induction_utils.set_plotting_style() in the preamble.
    """
    rc = {'lines.linewidth': 1.25,
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'axes.facecolor': '#E3DCD0',
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
        #   'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': '-',
          'grid.linewidth': 0.1,
          'grid.color': '#ffffff',
          'legend.fontsize': 9}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('xtick.major', pad=-1)
    plt.rc('ytick.major', pad=-1)
    plt.rc('mathtext', fontset='stixsans', sf='sansserif')
    plt.rc('figure', figsize=[3.5, 2.5])
    plt.rc('svg', fonttype='none')
    plt.rc('legend', title_fontsize='8', frameon=True, 
           facecolor='#E3DCD0', framealpha=1)
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)




######################################### raw data processing #########################################
    

def consolidate_images(base_dir):
    # Dynamically list all experimental folders using glob
    folders = [f for f in glob.glob(os.path.join(base_dir, '*/')) if os.path.isdir(f)]
    folders.sort()  # Optional: sort to ensure consistent processing order

    # Derive the new directory name from the common prefix
    common_prefix = os.path.commonprefix([os.path.basename(os.path.normpath(f)) for f in folders])
    new_dir = os.path.join(base_dir, common_prefix)

    # Check if the new directory already exists and has more than one subfolder
    if os.path.exists(new_dir) and len([f for f in os.listdir(new_dir) if os.path.isdir(os.path.join(new_dir, f))]) > 1:
        print(f"Consolidation appears to be already done. Directory '{new_dir}' already exists with subfolders.")
        return

    # Create the new directory for consolidated images
    os.makedirs(new_dir, exist_ok=True)

    # Dynamically list all Pos folders from the first experimental folder
    first_folder_path = folders[0]
    pos_folders = [d for d in os.listdir(first_folder_path) if os.path.isdir(os.path.join(first_folder_path, d))]
    pos_folders.sort()  # Optional: sort to ensure consistent processing order

    # Create subfolders for each Pos
    for pos in pos_folders:
        pos_folder_path = os.path.join(new_dir, pos)
        os.makedirs(pos_folder_path, exist_ok=True)

    # Initialize counters for Cy5, GFP, DAPI, Brightfield images, metadata, and display_and_comments.txt files
    cy5_counter = {pos: 0 for pos in pos_folders}
    gfp_counter = {pos: 0 for pos in pos_folders}
    dapi_counter = {pos: 0 for pos in pos_folders}
    brightfield_counter = {pos: 0 for pos in pos_folders}  # New brightfield counter
    metadata_counter = {pos: 0 for pos in pos_folders}
    comments_counter = 0

    # Function to count images in a folder
    def count_images(folder_path):
        if not os.path.exists(folder_path):
            return 0
        return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    # Move images, metadata, and display_and_comments.txt, and update counters
    for folder in folders:
        for pos in pos_folders:
            current_pos_folder_path = os.path.join(folder, pos)
            
            if not os.path.exists(current_pos_folder_path):
                print(f"Warning: {current_pos_folder_path} does not exist.")
                continue
            
            images = sorted(os.listdir(current_pos_folder_path))
            
            for image in images:
                old_image_path = os.path.join(current_pos_folder_path, image)
                image_lower = image.lower()  # Make the image name lowercase
                
                if 'cy5' in image_lower:
                    prefix, ext = os.path.splitext(image)
                    parts = prefix.split('_')
                    new_image_name = f'img_{cy5_counter[pos]:09d}_{parts[2]}_{parts[3]}{ext}'
                    cy5_counter[pos] += 1
                elif 'gfp' in image_lower:
                    prefix, ext = os.path.splitext(image)
                    parts = prefix.split('_')
                    new_image_name = f'img_{gfp_counter[pos]:09d}_{parts[2]}_{parts[3]}{ext}'
                    gfp_counter[pos] += 1
                elif 'dapi' in image_lower:
                    prefix, ext = os.path.splitext(image)
                    parts = prefix.split('_')
                    new_image_name = f'img_{dapi_counter[pos]:09d}_{parts[2]}_{parts[3]}{ext}'
                    dapi_counter[pos] += 1
                elif 'brightfield' in image_lower:  # New brightfield handling
                    prefix, ext = os.path.splitext(image)
                    parts = prefix.split('_')
                    new_image_name = f'img_{brightfield_counter[pos]:09d}_{parts[2]}_{parts[3]}{ext}'
                    brightfield_counter[pos] += 1
                elif image == 'metadata.txt':
                    # Move and rename metadata.txt to avoid overwriting
                    new_image_name = f'metadata_{metadata_counter[pos]:03d}.txt'
                    metadata_counter[pos] += 1
                else:
                    continue
                
                new_image_path = os.path.join(new_dir, pos, new_image_name)
                
                try:
                    shutil.move(old_image_path, new_image_path)
                except Exception as e:
                    print(f"Error moving {old_image_path} to {new_image_path}: {e}")

            # Move display_and_comments.txt and rename it
            comments_file_path = os.path.join(folder, 'display_and_comments.txt')
            if os.path.exists(comments_file_path):
                new_comments_path = os.path.join(new_dir, f'display_and_comments_{comments_counter:03d}.txt')
                comments_counter += 1
                try:
                    shutil.move(comments_file_path, new_comments_path)
                except Exception as e:
                    print(f"Error moving {comments_file_path} to {new_comments_path}: {e}")

            # If the Pos folder is empty after moving images, delete it
            if not os.listdir(current_pos_folder_path):
                try:
                    os.rmdir(current_pos_folder_path)
                    print(f"Deleted empty folder: {current_pos_folder_path}")
                except Exception as e:
                    print(f"Error deleting folder {current_pos_folder_path}: {e}")

        # If the main folder is empty after moving the display_and_comments.txt file, delete it
        if not os.listdir(folder):
            try:
                os.rmdir(folder)
                print(f"Deleted empty folder: {folder}")
            except Exception as e:
                print(f"Error deleting folder {folder}: {e}")

    # Check and count images in final folders
    print("\nChecking final consolidated folders:")
    for pos in pos_folders:
        pos_folder_path = os.path.join(new_dir, pos)
        count = count_images(pos_folder_path)
        print(f"Images in final {pos_folder_path}: {count}")

    print("Renaming, moving, and cleanup completed.")



def organize_conditions(data_path, conditions_dict):
    """
    Organizes PosX folders into condition folders as specified by conditions_dict.
    
    Args:
        data_path (str): Path to the data directory.
        conditions_dict (dict): Dictionary where keys are condition names and values are lists of PosX folders.
    """

    

    for condition, pos_folders in conditions_dict.items():
        # Create condition folder if it doesn't exist
        condition_path = os.path.join(data_path, condition)
        os.makedirs(condition_path, exist_ok=True)
        
        # Ensure pos_folders is a list, even if only one PosX is provided
        if isinstance(pos_folders, str):
            pos_folders = [pos_folders]
        
        # Move PosX folders into the condition folder
        for pos_folder in pos_folders:
            src_path = os.path.join(data_path, pos_folder)
            dest_path = os.path.join(condition_path, pos_folder)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
            else:
                print(f"Warning: {src_path} does not exist. Skipping.")


def prepare_conditions(data_path):
    """
    Prepares conditions and subconditions, renaming subconditions to 'RepX'.
    
    Args:
        data_path (str): Path to the data directory.
    
    Returns:
        conditions (list): List of condition names.
        subconditions (list): List of renamed subconditions as 'RepX'.
    """
    # List conditions while ignoring 'output_data'
    conditions = natsorted([
        f for f in os.listdir(data_path) 
        if os.path.isdir(os.path.join(data_path, f)) and f != 'output_data'
    ])
    
    # Determine the maximum number of subconditions across all conditions
    max_num_subconditions = max([
        len([
            f for f in os.listdir(os.path.join(data_path, condition)) 
            if os.path.isdir(os.path.join(data_path, condition, f))
        ])
        for condition in conditions
    ])
    
    # Rename subconditions to 'RepX' where X is the index (1-based)
    subconditions = [f'Rep{i+1}' for i in range(max_num_subconditions)]
    
    return conditions, subconditions


def reorgTiffsToOriginal(data_path, conditions, subconditions):
    """
    Renames subconditions as RepX and moves the raw data to the "original" folder.
    
    Args:
        data_path (str): Path to the data directory.
        conditions (list): List of conditions.
        subconditions (list): List of subconditions.
    """
    for condition in conditions:
        # Get the actual subconditions in the directory
        actual_subconditions = [name for name in os.listdir(os.path.join(data_path, condition)) if os.path.isdir(os.path.join(data_path, condition, name))]
        
        # Ensure subconditions list matches the number of actual subconditions
        actual_subconditions.sort()
        matched_subconditions = subconditions[:len(actual_subconditions)]
        
        # Rename the actual subconditions to match the subconditions in your list
        for i, actual_subcondition in enumerate(actual_subconditions):
            os.rename(os.path.join(data_path, condition, actual_subcondition), os.path.join(data_path, condition, matched_subconditions[i]))
        
        for subcondition in matched_subconditions:
            # Construct the path to the subcondition directory
            subcondition_path = os.path.join(data_path, condition, subcondition)
            
            # Create the path for the "original" directory within the subcondition directory
            original_dir_path = os.path.join(subcondition_path, "original")
            
            # Always create the "original" directory
            os.makedirs(original_dir_path, exist_ok=True)
            
            # Iterate over all files in the subcondition directory
            for filename in os.listdir(subcondition_path):
                # Check if the file is a .tif file
                if filename.endswith(".tif"):
                    # Construct the full path to the file
                    file_path = os.path.join(subcondition_path, filename)
                    
                    # Construct the path to move the file to
                    destination_path = os.path.join(original_dir_path, filename)
                    
                    # Move the file to the "original" directory
                    shutil.move(file_path, destination_path)
            print(f"Moved .tif files from {subcondition_path} to {original_dir_path}")




######################################### Video Creation #########################################



def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def calculate_mean_intensity(path):
    """Calculate mean intensity of an image within a 730x730 radius circle in the center."""
    img = io.imread(path)
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    radius = 730 // 2

    # Create a mask for the circle
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius

    # Apply the mask to the image
    circular_region = img[mask]
    circular_region = circular_region.mean()

    # calculate the mean intensity of the entire image
    mean_intensity = img.mean()

    # Return mean intensity of the circular region
    return mean_intensity

def calculate_protein_concentration_ug_ml(mean_intensity, intercept, slope):
    """Calculate protein concentration in ng/ul and nM."""
    conc_ug_ml = (mean_intensity - intercept) / slope
    return conc_ug_ml

def calculate_protein_concentration_nM(conc_ug_ml):
    """Convert protein concentration from ng/ul to nM."""
    conc_nM = conc_ug_ml * 1E-3 / 27000 * 1E9 
    return conc_nM

def calculate_number_of_protein_molecules(conc_nM):
    """Calculate number of protein molecules in 1µl given a concentration in nM."""
    avogadro_number = 6.022e23  # Avogadro's number (molecules per mole)
    conc_moles_per_l = conc_nM * 1e-9  # Convert concentration from nM to moles per liter
    conc_moles_per_ul = conc_moles_per_l * 1e-6  # Convert concentration from moles per liter to moles per microliter
    number_of_molecules = conc_moles_per_ul * avogadro_number  # Calculate number of molecules in 1µl
    return number_of_molecules

def convert_time_units(time_values_s):
    """Convert time values from seconds to minutes and hours."""
    time_values_min = time_values_s / 60
    time_values_h = time_values_s / 3600
    return time_values_s, time_values_min, time_values_h

def process_image(args):
    import matplotlib.patheffects as patheffects  # For scale bar text outline

    # Unpack arguments, allowing for an optional custom_title argument
    # If args has 14 elements, the last is custom_title; otherwise, custom_title is None/False
    if len(args) == 14:
        (image_file, output_directory_path, channel, slope, intercept, vmax, time_interval, i, show_scalebar, min_frame, skip_frames, condition, subcondition, custom_title) = args
    else:
        (image_file, output_directory_path, channel, slope, intercept, vmax, time_interval, i, show_scalebar, min_frame, skip_frames, condition, subcondition) = args
        custom_title = False

    # Read the image into a numpy array
    intensity_matrix = io.imread(image_file)

    if channel == "cy5":
        # Normalize intensity matrix to range [0, 1] for cy5 channel
        matrix_to_plot = intensity_matrix / 1000
        label = 'Normalized Fluorescence Intensity'
    else:
        # Convert intensity values to protein concentration using the calibration curve
        matrix_to_plot = calculate_protein_concentration_ug_ml(intensity_matrix, slope, intercept)
        matrix_to_plot = matrix_to_plot / 27000 * 1E6
        label = 'Protein concentration (nM)'

    # Plot the heatmap with a larger figure size
    fig, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(matrix_to_plot, cmap='gray', interpolation='nearest', vmin=0, vmax=vmax)

    if show_scalebar:
        plt.colorbar(im, ax=ax, label=label)
    
    # Remove axes and make image fill whole size
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add title inside the image at top right, but ensure it doesn't go out of bounds
    if custom_title:
        title_text = str(custom_title)
    else:
        title_text = f"{condition}"

    # Truncate the title if it's too long to fit in the image
    # Estimate max characters based on font size and figure width
    # This is a heuristic: adjust max_chars as needed for your use case
    max_chars = 30  # Reduced for right alignment
    if len(title_text) > max_chars:
        title_text = title_text[:max_chars-3] + "..."

    # Place the title at top right with right alignment
    # Use 0.98 for x position to give some margin from the right edge
    ax.text(0.98, 0.98, title_text, 
            transform=ax.transAxes, color='white', fontsize=40, 
            weight='bold', va='top', ha='right',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black', alpha=0.7)],
            clip_on=True)
    
    # Add timer information at top left in white
    time_hours = (i - min_frame) * time_interval * skip_frames / 3600
    time_minutes = (i - min_frame) * time_interval * skip_frames / 60
    
    ax.text(0.02, 0.99, f"{time_hours:.2f} h", 
            transform=ax.transAxes, color='white', fontsize=38, 
            weight='bold', va='top', ha='left',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black', alpha=0.7)])
    
    ax.text(0.02, 0.94, f"{time_minutes:.2f} min", 
            transform=ax.transAxes, color='white', fontsize=38, 
            weight='bold', va='top', ha='left',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='black', alpha=0.7)])

    # Draw a 1mm scale bar (730 pixels) at the bottom right
    scalebar_length_px = 730
    scalebar_height = max(4, int(matrix_to_plot.shape[0] * 0.005))  # 4 pixels or 0.5% of image height
    color = 'white' if np.mean(matrix_to_plot) < 0.5 * vmax else 'black'

    # Coordinates for the scale bar
    x_start = matrix_to_plot.shape[1] - scalebar_length_px - 40  # 40 px from right edge
    x_end = matrix_to_plot.shape[1] - 40
    y_pos = matrix_to_plot.shape[0] - 40  # 40 px from bottom

    # Draw the scale bar as a thick line
    ax.hlines(
        y=y_pos, xmin=x_start, xmax=x_end, colors=color, linewidth=scalebar_height, zorder=10, alpha=0.9
    )
    # Add text label above the scale bar
    ax.text(
        (x_start + x_end) / 2, y_pos - 15, "1 mm", color=color, fontsize=18, ha='center', va='bottom', weight='bold', zorder=11,
        path_effects=[patheffects.withStroke(linewidth=3, foreground='black' if color == 'white' else 'white', alpha=0.7)]
    )

    # Save the heatmap with no borders
    heatmap_filename = f"heatmap_frame_{i}.png"
    heatmap_path = os.path.join(output_directory_path, heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)

def fluorescence_heatmap(data_path, conditions, subconditions, channel, time_interval_list, vmax, min_frame=0, max_frame=None, skip_frames=1, calibration_curve_paths=None, show_scalebar=True, batch_size=100, custom_title=None):
    """
    Reads each image as a matrix, creates, and saves a heatmap representing the normalized pixel-wise fluorescence intensity.

    Args:
    - data_path (str): Base directory where the images are stored.
    - conditions (list): List of conditions defining subdirectories within the data path.
    - subconditions (list): List of subconditions defining further subdirectories.
    - channel (str): Channel specifying the fluorescence ('cy5' or 'gfp').
    - time_interval_list (list): List of time intervals in seconds between frames for each condition.
    - min_frame (int): Minimum frame number to start processing from.
    - max_frame (int): Maximum frame number to stop processing at.
    - vmax (float | list | dict): Maximum value(s) for color scale in the heatmap.
        If float, applied to all movies. If list, it must have the same length as
        subconditions and is applied by position to each subcondition for all conditions.
        If dict, keys can be subcondition names or "condition:subcondition" strings
        to target specific movies.
    - skip_frames (int): Interval to skip frames (default is 1, meaning process every frame).
    - calibration_curve_paths (list): List of file paths for the calibration curve images.
    - show_scalebar (bool): Whether to show the color scale bar in the heatmap.
    - batch_size (int): Number of images to process in each batch to avoid memory overload.
    - custom_title (str or None): Custom title to display on the heatmap. If None or False, use default.
    """
    output_data_dir = os.path.join(data_path, "output_data", "movies")
    ensure_output_dir(output_data_dir)

    for idx, condition in enumerate(conditions):
        time_interval = time_interval_list[idx]

        for sub_idx, subcondition in enumerate(subconditions):
            # Determine the directory paths based on the channel
            input_directory_path = os.path.join(data_path, condition, subcondition, "original")
            output_directory_path = os.path.join(output_data_dir, f"{condition}_{subcondition}_heatmaps_{channel}")

            # Create the output directory if it doesn't exist, or clear it if it does
            if os.path.exists(output_directory_path):
                shutil.rmtree(output_directory_path)
            os.makedirs(output_directory_path, exist_ok=True)

            # Get all .tif files in the folder
            image_files = sorted(glob.glob(os.path.join(input_directory_path, f"*{channel}*.tif")))[min_frame:max_frame:skip_frames]

            # Setup calibration curve for non-cy5 channels
            slope, intercept = None, None
            if channel != "cy5":
                # Calibration curve data and fit
                initial_concentration = 285 
                sample_concentration_values = [initial_concentration/64, initial_concentration/32, initial_concentration/16, initial_concentration/8, initial_concentration/4, initial_concentration/2, ]

                if calibration_curve_paths is None or len(calibration_curve_paths) != len(sample_concentration_values):
                    raise ValueError(f"Mismatch in lengths: {len(calibration_curve_paths)} calibration images, {len(sample_concentration_values)} sample concentrations")

                mean_intensity_calibration = [calculate_mean_intensity(path) for path in calibration_curve_paths]
                slope, intercept = np.polyfit(sample_concentration_values, mean_intensity_calibration, 1)

            # Progress bar for the entire subcondition
            with tqdm(total=len(image_files), desc=f"Processing {condition} - {subcondition}", leave=True, dynamic_ncols=True) as pbar:
                # Process images in batches to avoid memory overload
                for batch_start in range(0, len(image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_files))
                    batch_files = image_files[batch_start:batch_end]

                    # Resolve vmax for this specific movie (condition/subcondition)
                    if isinstance(vmax, dict):
                        vmax_key_specific = f"{condition}:{subcondition}"
                        vmax_for_movie = vmax.get(vmax_key_specific, vmax.get(subcondition, None))
                        if vmax_for_movie is None:
                            # Fall back to any 'default' key or raise if not provided
                            vmax_for_movie = vmax.get("default", None)
                            if vmax_for_movie is None:
                                raise ValueError(f"vmax dict missing key for '{vmax_key_specific}' or '{subcondition}' and no 'default' provided.")
                    elif isinstance(vmax, (list, tuple)):
                        if len(vmax) != len(subconditions):
                            raise ValueError(f"Length of vmax list ({len(vmax)}) must equal number of subconditions ({len(subconditions)}).")
                        vmax_for_movie = vmax[sub_idx]
                    else:
                        vmax_for_movie = vmax

                    # Prepare arguments for multiprocessing
                    if custom_title:
                        args = [(image_file, output_directory_path, channel, slope, intercept, vmax_for_movie, time_interval, i, show_scalebar, min_frame, skip_frames, condition, subcondition, custom_title)
                                for i, image_file in enumerate(batch_files, start=batch_start + min_frame)]
                    else:
                        args = [(image_file, output_directory_path, channel, slope, intercept, vmax_for_movie, time_interval, i, show_scalebar, min_frame, skip_frames, condition, subcondition)
                                for i, image_file in enumerate(batch_files, start=batch_start + min_frame)]

                    with mp.Pool(mp.cpu_count()) as pool:
                        for _ in pool.imap(process_image, args):
                            pbar.update(1)



def process_video_creation(args):
    image_files, out_path, frame_rate = args

    # Get the resolution of the first image (assuming all images are the same size)
    first_image = cv2.imread(image_files[0])
    video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, frame_rate, video_resolution)

    for file in image_files:
        img = cv2.imread(file)
        out.write(img)  # Write the image as a frame in the video

    out.release()

def create_movies(data_path, conditions, subconditions, channel, frame_rate=30, max_frame=None, skip_frames=1, batch_size=100):
    """
    Creates video files from heatmaps stored in the specified directory.

    Args:
    - data_path (str): Base path where the heatmaps are stored.
    - conditions (list): List of conditions defining subdirectories within the data path.
    - subconditions (list): List of subconditions defining further subdirectories.
    - channel (str): The specific channel being processed ('cy5' or 'gfp').
    - frame_rate (int): Frame rate for the output video. Defaults to 30.
    - max_frame (int, optional): Maximum number of frames to be included in the video. If None, all frames are included.
    - skip_frames (int): Interval to skip frames (default is 1, meaning process every frame).
    - batch_size (int): Number of images to process in each batch to avoid memory overload.
    """
    output_data_dir = os.path.join(data_path, "output_data", "movies")
    ensure_output_dir(output_data_dir)

    for condition in conditions:
        for subcondition in subconditions:
            images_dir = os.path.join(output_data_dir, f"{condition}_{subcondition}_heatmaps_{channel}")
            image_files = natsorted(glob.glob(os.path.join(images_dir, "*.png")))[::skip_frames]
            
            if max_frame is not None:
                image_files = image_files[:max_frame]

            if len(image_files) == 0:
                print(f"No images found for {condition} - {subcondition} in {channel}.")
                continue

            # Calculate the video duration
            video_duration = len(image_files) / frame_rate
            print(f"Creating video for {condition} - {subcondition} with duration: {video_duration:.2f} seconds.")
            
            # Create a filename including frame rate and total frame count
            video_filename = f"{condition}_{subcondition}_{channel}_{frame_rate}fps_{len(image_files)}frames.avi"
            out_path = os.path.join(output_data_dir, video_filename)

            # Get the resolution of the first image (assuming all images are the same size)
            first_image = cv2.imread(image_files[0])
            video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

            # Define the codec and create the VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_path, fourcc, frame_rate, video_resolution)

            # Progress bar for the entire subcondition
            with tqdm(total=len(image_files), desc=f"Creating video for {condition} - {subcondition}", leave=True, dynamic_ncols=True) as pbar:
                # Process images in batches to avoid memory overload
                for batch_start in range(0, len(image_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_files))
                    batch_files = image_files[batch_start:batch_end]

                    for image_file in batch_files:
                        img = cv2.imread(image_file)
                        out.write(img)  # Write the image as a frame in the video
                        pbar.update(1)

            # Release the video writer
            out.release()


def process_frame(args):
    frame_index, temp_img_dir, conditions, subconditions, channel, grid_rows, grid_cols, data_path = args
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 12, grid_rows * 12))
    plt.subplots_adjust(hspace=0, wspace=0)  # No spacing between movies

    # Ensure axes is always 2D
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1:
        axes = np.array(axes).reshape(grid_rows, grid_cols)

    plot_index = 0

    # Loop through each condition and subcondition
    for col_idx, condition in enumerate(conditions):
        for row_idx, subcondition in enumerate(subconditions):
            # Determine the image path
            images_dir = os.path.join(data_path, "output_data", "movies", f"{condition}_{subcondition}_heatmaps_{channel}")
            image_files = natsorted(glob.glob(os.path.join(images_dir, "*.png")))

            if frame_index < len(image_files):
                # Use the available frame
                image_path = image_files[frame_index]
            else:
                # If no more frames, use the last available frame
                image_path = image_files[-1]

            img = io.imread(image_path)

            # Plot the image in the appropriate subplot
            ax = axes[row_idx if len(subconditions) > 1 else plot_index // grid_cols,
                      col_idx if len(subconditions) > 1 else plot_index % grid_cols]
            ax.imshow(img, cmap='gray', vmin=0, vmax=img.max())
            ax.axis('off')  # Remove axes
            
            # Remove all spines to eliminate any border lines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plot_index += 1

    # Turn off any unused subplots
    for ax in axes.flatten()[plot_index:]:
        ax.axis('off')

    # Save the combined frame
    combined_image_path = os.path.join(temp_img_dir, f"combined_frame_{frame_index:04d}.png")
    plt.savefig(combined_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def create_combined_heatmap_movie_custom_grid(data_path, conditions, subconditions, channel, grid_rows=None, grid_cols=None, frame_rate=30, batch_size=50):
    """
    Combines heatmaps from different conditions and subconditions into a single video.
    Allows specifying the number of grid rows and columns or uses an adaptive layout based on subconditions.

    Args:
    - data_path (str): Base path where the heatmaps are stored.
    - conditions (list): List of conditions defining subdirectories within the data path.
    - subconditions (list): List of subconditions defining further subdirectories.
    - channel (str): The specific channel being processed ('cy5' or 'gfp').
    - grid_rows (int, optional): Number of rows in the grid. If None, calculated adaptively.
    - grid_cols (int, optional): Number of columns in the grid. If None, calculated adaptively.
    - frame_rate (int): Frame rate for the output video. Defaults to 30.
    - batch_size (int): Number of frames to process in each batch to avoid memory overload.
    """
    # Determine grid dimensions if not provided
    total_plots = len(conditions) * len(subconditions)
    
    if grid_rows is None or grid_cols is None:
        if len(subconditions) == 1:
            grid_cols = int(np.ceil(np.sqrt(total_plots)))
            grid_rows = int(np.ceil(total_plots / grid_cols))
            while grid_cols * grid_rows >= total_plots:
                if (grid_cols - 1) * grid_rows >= total_plots:
                    grid_cols -= 1
                elif grid_cols * (grid_rows - 1) >= total_plots:
                    grid_rows -= 1
                else:
                    break
        else:
            grid_rows = len(subconditions)
            grid_cols = len(conditions)
    
    # Define the output directory for temporary images (now called 'combined_frames' in 'movies' directory)
    output_data_dir = os.path.join(data_path, "output_data", "movies")
    combined_frames_dir = os.path.join(output_data_dir, "combined_frames")
    ensure_output_dir(combined_frames_dir)

    # Determine the maximum number of frames based on the longest video
    max_num_frames = 0
    for condition in conditions:
        for subcondition in subconditions:
            image_dir = os.path.join(data_path, "output_data", "movies", f"{condition}_{subcondition}_heatmaps_{channel}")
            num_frames = len(natsorted(glob.glob(os.path.join(image_dir, "*.png"))))
            if num_frames > max_num_frames:
                max_num_frames = num_frames

    if max_num_frames == 0:
        print(f"No frames to process. Check if the directories exist and contain images.")
        return

    # Calculate and print the video duration
    video_duration = max_num_frames / frame_rate
    print(f"Creating video with duration: {video_duration:.2f} seconds.")

    # Progress bar for the entire operation
    with tqdm(total=max_num_frames, desc="Creating combined frames", leave=True, dynamic_ncols=True) as pbar:
        for batch_start in range(0, max_num_frames, batch_size):
            batch_end = min(batch_start + batch_size, max_num_frames)
            batch_frames = range(batch_start, batch_end)

            args_list = [
                (
                    frame_index, combined_frames_dir, conditions, subconditions, channel,
                    grid_rows, grid_cols, data_path
                )
                for frame_index in batch_frames
            ]
            
            with mp.Pool(mp.cpu_count()) as pool:
                for _ in pool.imap(process_frame, args_list):
                    pbar.update(1)

    # Compile the images into a video using OpenCV
    combined_image_files = natsorted(glob.glob(os.path.join(combined_frames_dir, "combined_frame_*.png")))

    # Get the resolution of the first image
    first_image = cv2.imread(combined_image_files[0])
    height, width, layers = first_image.shape
    video_resolution = (width, height)

    # Define the codec and create a VideoWriter object
    output_filename = f"combined_heatmap_movie_{channel}_{frame_rate}fps_{max_num_frames}frames.avi"
    output_file = os.path.join(output_data_dir, output_filename)
    ensure_output_dir(output_data_dir)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, video_resolution)

    for image_file in combined_image_files:
        img = cv2.imread(image_file)
        out.write(img)  # Write the image as a frame in the video

    out.release()
    print(f"Combined video saved to {output_file}")


def delete_temporary_image_directories(data_path, conditions, subconditions):
    """
    Deletes all the temporary directories containing the images used for creating movies, for all channels.

    Args:
    - data_path (str): Base path where the temporary images are stored.
    - conditions (list): List of conditions defining subdirectories within the data path.
    - subconditions (list): List of subconditions defining further subdirectories.
    """
    # Define the output directory
    output_data_dir = os.path.join(data_path, "output_data", "movies")
    
    for condition in conditions:
        for subcondition in subconditions:
            # Find all channel-specific directories and remove them
            temp_dirs = glob.glob(os.path.join(output_data_dir, f"{condition}_{subcondition}_heatmaps_*"))
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Deleted temporary directory: {temp_dir}")
    
    # Delete the 'combined_frames' directory used in combined movie creation
    combined_frames_dir = os.path.join(output_data_dir, "combined_frames")
    if os.path.exists(combined_frames_dir):
        shutil.rmtree(combined_frames_dir)
        print(f"Deleted temporary images directory: {combined_frames_dir}")

def delete_produced_output_all_channels(output_base_dir, conditions, subconditions):
    """
    Deletes all the produced output including temporary directories and generated files for all channels.
    
    Args:
    - output_base_dir (str): The base directory where output files are stored.
    - conditions (list): List of conditions defining subdirectories within the output base directory.
    - subconditions (list): List of subconditions defining further subdirectories.
    """
    output_data_dir = os.path.join(output_base_dir, "output_data")
    
    # Delete the main output_data directory if it exists
    if os.path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)
        print(f"Deleted main output directory: {output_data_dir}")
    
    # Loop through each condition and subcondition to delete individual directories
    for condition in conditions:
        for subcondition in subconditions:
            # Remove all heatmap directories for each condition and subcondition
            channel_dirs = glob.glob(os.path.join(output_data_dir, f"movies/{condition}_{subcondition}_heatmaps_*"))
            for temp_dir in channel_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Deleted temporary directory: {temp_dir}")
                
            # Find and remove all combined movie files
            combined_movie_files = glob.glob(os.path.join(output_data_dir, f"movies/combined_heatmap_movie_*.avi"))
            for combined_movie_file in combined_movie_files:
                if os.path.exists(combined_movie_file):
                    os.remove(combined_movie_file)
                    print(f"Deleted combined movie file: {combined_movie_file}")

    # Delete the 'combined_frames' directory used for combined movie creation
    temp_img_dir = os.path.join(output_data_dir, "movies/combined_frames")
    if os.path.exists(temp_img_dir):
        shutil.rmtree(temp_img_dir)
        print(f"Deleted temporary images directory: {temp_img_dir}")



######################################### Fluorescence Quantification #########################################




def quantify_tiffiles(data_path, conditions, subconditions, calibration_curve_paths, droplet_volume_list, time_interval_s_list, skip_frames=1, ):
    """Process images to calculate protein concentration and generate plots, with an option to skip frames."""
    all_data = []

    # Sort the calibration curve paths
    calibration_curve_paths = sorted(calibration_curve_paths)

    # Calibration curve data and fit
    initial_concentration = 285 
    sample_concentration_values = [initial_concentration/64, initial_concentration/32, initial_concentration/16, initial_concentration/8, initial_concentration/4, initial_concentration/2, ]
    with mp.Pool(mp.cpu_count()) as pool:
        mean_intensity_calibration = pool.map(calculate_mean_intensity, calibration_curve_paths)
    slope, intercept = np.polyfit(sample_concentration_values, mean_intensity_calibration, 1)

    for idx, condition in enumerate(conditions):
        droplet_volume = droplet_volume_list[idx]
        time_interval_s = time_interval_s_list[idx]

        for subcondition in subconditions:
            pattern = os.path.join(data_path, condition, subcondition, "original", "*[Gg][Ff][Pp]*.tif")
            paths = sorted(glob.glob(pattern))

            if not paths:
                print(f"No image files found for condition {condition}, subcondition {subcondition}.")
                continue


            # Apply skip_frames in both cases
            paths = paths[::skip_frames]

            with mp.Pool(mp.cpu_count()) as pool:
                mean_intensity_list = list(tqdm(pool.imap(calculate_mean_intensity, paths), total=len(paths), desc=f"Calculating intensities for {condition} - {subcondition}"))
                

            protein_concentration_list = [calculate_protein_concentration_ug_ml(intensity, intercept, slope) for intensity in mean_intensity_list]
            protein_concentration_nM_list = [calculate_protein_concentration_nM(conc_ng_ul) for conc_ng_ul in protein_concentration_list]

            # min_intensity = min(mean_intensity_list)
            # mean_intensity_list = np.array(mean_intensity_list) - min_intensity
            protein_concentration_list = np.array(protein_concentration_list) - min(protein_concentration_list)
            protein_concentration_nM_list = np.array(protein_concentration_nM_list) - min(protein_concentration_nM_list)

            time_values_s = np.arange(len(mean_intensity_list)) * time_interval_s * skip_frames
            time_values_s, time_values_min, time_values_h = convert_time_units(time_values_s)
            
            protein_mass_list = protein_concentration_list * droplet_volume
            df = pd.DataFrame({
                "Condition": condition,
                "Subcondition": subcondition,
                "time (s)": time_values_s,
                "Time_min": time_values_min,
                "Time_h": time_values_h,
                "Mean Intensity": mean_intensity_list,
                "Protein Concentration_ng_ul": protein_concentration_list,
                "Protein Concentration_nM": protein_concentration_nM_list,
            })


            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    mean_df = combined_df.groupby(["Condition", "time (s)", "Time_min", "Time_h"]).mean(numeric_only=True).reset_index()


    # Reset index of both dataframes 
    combined_df.reset_index(drop=True)
    mean_df.reset_index(drop=True)

    output_dir = os.path.join(data_path, "output_data")
    ensure_output_dir(output_dir)

    combined_csv_path = os.path.join(output_dir, "combined_expression.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    mean_csv_path = os.path.join(output_dir, "mean_expression.csv")
    mean_df.to_csv(mean_csv_path, index=False)

    plot_results(combined_df, mean_df, output_dir, sample_concentration_values, mean_intensity_calibration, slope, intercept, )

    return combined_csv_path, mean_csv_path




def plot_results(df, mean_df, output_dir, sample_concentration_values, mean_intensity_calibration, slope, intercept, ):
    """Generate and save plots for the experimental data."""
    
    calibration_dir = os.path.join(output_dir, "calibration")
    ensure_output_dir(calibration_dir)
    
    combined_plots_dir = os.path.join(output_dir, "combined_expression_plots")
    ensure_output_dir(combined_plots_dir)
    
    mean_plots_dir = os.path.join(output_dir, "mean_expression_plots")
    ensure_output_dir(mean_plots_dir)
    
    combined_log_plots_dir = os.path.join(output_dir, "combined_expression_plots_log")
    ensure_output_dir(combined_log_plots_dir)
    
    mean_log_plots_dir = os.path.join(output_dir, "mean_expression_plots_log")
    ensure_output_dir(mean_log_plots_dir)
    
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["image.cmap"] = "viridis"
    dpi_setting = 200
    
    # Plot calibration curve
    plt.figure(dpi=dpi_setting)
    plt.scatter(sample_concentration_values, mean_intensity_calibration, label="Data Points")
    plt.plot(sample_concentration_values, np.polyval([slope, intercept], sample_concentration_values), color='r', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("Protein Concentration ng_ul")
    plt.ylabel("Mean Intensity")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(calibration_dir, "calibration_curve.png"), dpi=dpi_setting)
    plt.close()

    # Determine time units from dataframe
    time_units = [(col, col.replace('_', ' ').title()) for col in df.columns if col.startswith("Time_")]
    
    # Determine metrics (columns to plot) dynamically, excluding time units and non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    metrics = [(col, col.replace('_', ' ').title()) for col in numeric_cols if col not in [col[0] for col in time_units]]

    # Plot combined data for each metric
    for metric, ylabel in metrics:
        for time_unit, xlabel in time_units:
            plt.figure(dpi=dpi_setting)
            for (condition, subcondition), group in df.groupby(["Condition", "Subcondition"]):
                plt.plot(group[time_unit], group[metric], label=f"{condition} - {subcondition}")
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Combined {ylabel} over {xlabel} for All Conditions")
            if metric == 'Translation Rate aa_s':
                plt.ylim(0, 10)  # Set y-axis limit for translation rate plot
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(combined_plots_dir, f"combined_{metric}_plot_{time_unit}.png"), dpi=dpi_setting)
            plt.close()

            # Generate log scale version
            plt.figure(dpi=dpi_setting)
            for (condition, subcondition), group in df.groupby(["Condition", "Subcondition"]):
                plt.plot(group[time_unit], group[metric], label=f"{condition} - {subcondition}")
            
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Combined {ylabel} over {xlabel} for All Conditions (Log Scale)")
            plt.yscale('log')
            if metric == 'Translation Rate aa_s':
                plt.ylim(1e-3, 1)  # Adjust y-axis limits for log scale if necessary
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(combined_log_plots_dir, f"combined_{metric}_plot_{time_unit}_log.png"), dpi=dpi_setting)
            plt.close()

    # Plot mean data for each metric
    for metric, ylabel in metrics:
        for time_unit, xlabel in time_units:
            plt.figure(dpi=dpi_setting)
            for condition, group in mean_df.groupby("Condition"):
                plt.plot(group[time_unit], group[metric], label=f"{condition}")
                
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Mean {ylabel} over {xlabel} for All Conditions")
            if metric == 'Translation Rate aa_s':
                plt.ylim(0, 1)  # Set y-axis limit for translation rate plot
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(mean_plots_dir, f"mean_{metric}_plot_{time_unit}.png"), dpi=dpi_setting)
            plt.close()

            # Generate log scale version
            plt.figure(dpi=dpi_setting)
            for condition, group in mean_df.groupby("Condition"):
                plt.plot(group[time_unit], group[metric], label=f"{condition}")
                
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Mean {ylabel} over {xlabel} for All Conditions (Log Scale)")
            plt.yscale('log')
            if metric == 'Translation Rate aa_s':
                plt.ylim(1e-3, 1)  # Adjust y-axis limits for log scale if necessary
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(mean_log_plots_dir, f"mean_{metric}_plot_{time_unit}_log.png"), dpi=dpi_setting)
            plt.close()



######################################### PIV pre-processing #########################################


def split_tiffs(data_path, conditions, subconditions, channel, file_interval=None):

    for condition in conditions:
        for subcondition in subconditions:
            # Construct the path to the 'original' directory within the subcondition
            original_dir_path = os.path.join(data_path, condition, subcondition, "original")

            if not os.path.exists(original_dir_path):
                print(f"Error: The original directory {original_dir_path} does not exist.")
                continue

            # Create the directory for the channel
            data_dir = os.path.join(data_path, condition, subcondition, f"{channel}-{file_interval}x")
            os.makedirs(data_dir, exist_ok=True)

            # Check if the expected output is already there
            expected_files = [f for f in sorted(os.listdir(original_dir_path))
                              if f.lower().endswith(".tif") and f"{channel}" in f.lower()]
            expected_output_files = expected_files[::file_interval or 1]
            already_copied_files = set(os.listdir(data_dir))

            # If all expected files are already copied, skip this subcondition
            if all(file in already_copied_files for file in expected_output_files):
                print(f"Skipping {subcondition} as the expected output is already present.")
                continue

            # Separate list for DAPI files
            dapi_files = []

            # Iterate over all files in the original directory
            file_list = sorted(os.listdir(original_dir_path))
            for filename in file_list:
                # Check if the file is a .tif file and contains 'DAPI' (case insensitive)
                if filename.lower().endswith(".tif") and f"{channel}" in filename.lower():
                    dapi_files.append(filename)

            # Copy files based on the file_interval
            if file_interval is None:
                file_interval = 1  # Copy all files if no interval is set

            for idx, filename in enumerate(dapi_files):
                if idx % file_interval == 0:
                    file_path = os.path.join(original_dir_path, filename)
                    shutil.copy(file_path, os.path.join(data_dir, filename))

            print(f"Copied every {file_interval}th f'{channel}' file from {original_dir_path} into {data_dir}.")



def split_tiffs_stack(data_path, conditions, subconditions, channels, file_interval=None):

    for condition in conditions:
        for subcondition in subconditions:
            # Construct the path to the 'original' directory within the subcondition
            original_dir_path = os.path.join(data_path, condition, subcondition, "original")

            if not os.path.exists(original_dir_path):
                print(f"Error: The original directory {original_dir_path} does not exist.")
                continue

            for channel in channels:
                # Create the directory for the channel
                data_dir = os.path.join(data_path, condition, subcondition, f"{channel}-{file_interval}x")
                os.makedirs(data_dir, exist_ok=True)

                # Check if the expected output is already there
                expected_files = [f for f in sorted(os.listdir(original_dir_path))
                                  if f.lower().endswith(".tif") and f"{channel}" in f.lower()]
                expected_output_files = expected_files[::file_interval or 1]
                already_copied_files = set(os.listdir(data_dir))

                # If all expected files are already copied, skip this subcondition
                if all(file in already_copied_files for file in expected_output_files):
                    print(f"Skipping {subcondition} for channel {channel} as the expected output is already present.")
                    continue

                # Separate list for channel-specific files
                channel_files = []

                # Iterate over all files in the original directory
                file_list = sorted(os.listdir(original_dir_path))
                for filename in file_list:
                    # Check if the file is a .tif file and contains the channel (case insensitive)
                    if filename.lower().endswith(".tif") and f"{channel}" in filename.lower():
                        channel_files.append(filename)

                # Copy files based on the file_interval
                if file_interval is None:
                    file_interval = 1  # Copy all files if no interval is set

                for idx, filename in enumerate(channel_files):
                    if idx % file_interval == 0:
                        file_path = os.path.join(original_dir_path, filename)
                        shutil.copy(file_path, os.path.join(data_dir, filename))

                print(f"Copied every {file_interval}th '{channel}' file from {original_dir_path} into {data_dir}.")


######################################### PIV #########################################


# Convert a single image (helper function for multiprocessing)
def process_single_image(file_name, output_dir, brightness_factor, contrast_factor, num_digits, i):
    image = Image.open(file_name).convert("L")
    image_resized = image.resize((2048, 2048), Image.LANCZOS)

    enhancer = ImageEnhance.Brightness(image_resized)
    image_brightened = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image_brightened)
    image_contrasted = enhancer.enhance(contrast_factor)

    padded_index = str(i + 1).zfill(num_digits)
    base_file_name = f'converted_image_{padded_index}.tif'
    processed_image_path = os.path.join(output_dir, base_file_name)
    image_contrasted.save(processed_image_path, format='TIFF', compression='tiff_lzw')


# Convert PIVlab images to the right size using multiprocessing
def convert_images(data_path, conditions, subconditions, max_frame, brightness_factor=1, contrast_factor=1, skip_frames=1):
    for condition in tqdm(conditions, desc="Conditions", leave=False):
        for subcondition in tqdm(subconditions, desc="Subconditions", leave=False):
            input_dir = os.path.join(data_path, condition, subcondition, "piv_movie")
            output_dir = os.path.join(data_path, condition, subcondition, "piv_movie_converted")

            os.makedirs(output_dir, exist_ok=True)

            input_files = natsorted(glob.glob(os.path.join(input_dir, '*.jpg')))

            if max_frame:
                input_files = input_files[:max_frame]

            # Apply frame skipping
            input_files = input_files[::skip_frames]

            output_files = natsorted(glob.glob(os.path.join(output_dir, '*.tif')))
            if len(input_files) <= len(output_files):
                print(f"Conversion might already be completed or partial for {output_dir}. Continuing...")
                # Optional: Add logic to check and continue incomplete work.

            num_digits = len(str(len(input_files)))

            # Use all available cores
            with Pool(cpu_count()) as pool:
                list(tqdm(pool.starmap(process_single_image, [(file_name, output_dir, brightness_factor, contrast_factor, num_digits, i) for i, file_name in enumerate(input_files)]), total=len(input_files), desc="Converting Images", leave=False))


# Helper function to plot autocorrelation
def plot_autocorrelation_values(data_path, condition, subcondition, frame_id, lambda_tau, results, fitted_values, intervector_distance_microns, time_interval):
    output_directory_dfs = os.path.join(data_path, condition, subcondition, "autocorrelation_plots")
    os.makedirs(output_directory_dfs, exist_ok=True)

    plt.figure(figsize=(10, 6))

    x_values = np.arange(len(results)) * intervector_distance_microns * 1E6

    plt.plot(x_values, results, label='Autocorrelation Values', marker='o', linestyle='-', markersize=5)
    plt.plot(x_values, fitted_values, label='Fitted Exponential Decay', linestyle='--', color='red')
    plt.axvline(x=lambda_tau, color='green', linestyle='-.', label=f'Correlation Length = {lambda_tau:.2f} µm')

    plt.xlabel('Scaled Lag (µm)')
    plt.ylabel('Autocorrelation')

    # Convert frame_id to time in minutes
    time_in_minutes = (frame_id * time_interval) / 60
    plt.title(f'Autocorrelation Function and Fitted Exponential Decay (Time: {time_in_minutes:.2f} min)')
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    filename = os.path.join(output_directory_dfs, f'autocorrelation_frame_{frame_id}.jpg')
    plt.savefig(filename, dpi=200, format='jpg')
    plt.close()




def correlation_length(data_frame):
    # Reshaping the data frame to 2D grids for x and y velocity components
    vx = data_frame.pivot(index='y [m]', columns='x [m]', values="u [m/s]").values
    vy = data_frame.pivot(index='y [m]', columns='x [m]', values="v [m/s]").values
    
    # Subtract mean (centering the data)
    vx -= np.mean(vx)
    vy -= np.mean(vy)
    
    # Combine x and y velocity components into a single complex velocity field
    # This allows us to capture both magnitude and direction in the FFT
    velocity_field = vx + 1j * vy

    # FFT to find the power spectrum and compute the autocorrelation
    fft_v = np.fft.fft2(velocity_field)
    autocorr = np.fft.ifft2(fft_v * np.conj(fft_v))
    autocorr = np.real(autocorr) / np.max(np.real(autocorr))  # Normalize the autocorrelation

    # Preparing to extract the autocorrelation values along the diagonal
    r_values = min(vx.shape) // 2
    results = np.zeros(r_values)
    for r in range(r_values):
        # Properly average over symmetric pairs around the center
        autocorrelation_value = (autocorr[r, r] + autocorr[-r, -r]) / 2
        results[r] = autocorrelation_value

    # Normalize the results to start from 1
    results /= results[0]

    # Remove NaN or inf values
    valid_indices = np.isfinite(results)
    valid_results = results[valid_indices]
    valid_distances = np.arange(len(results))[valid_indices]

    # If there are no valid points, return NaN
    if len(valid_results) == 0:
        return np.nan, results, np.nan, np.nan

    # Exponential decay fitting to extract the correlation length
    def exponential_decay(x, A, B, C):
        return A * np.exp(-x / B) + C

    # Fit parameters and handling potential issues with initial parameter guesses
    try:
        params, _ = curve_fit(exponential_decay, valid_distances, valid_results, p0=(1, 10, 0), maxfev=5000)
    except RuntimeError:
        # Handle cases where the curve fit does not converge
        params = [np.nan, np.nan, np.nan]  # Use NaN to indicate the fit failed

    A, B, C = params
    fitted_values = exponential_decay(np.arange(r_values), *params)

    # Calculate the correlation length
    intervector_distance_microns = ((data_frame["y [m]"].max() - data_frame["y [m]"].min()) / vx.shape[0])
    if B > 0 and A != C:  # Ensure valid values for logarithmic calculation
        lambda_tau = -B * np.log((0.3 - C) / A) * intervector_distance_microns
    else:
        lambda_tau = np.nan  # Return NaN if parameters are not suitable for calculation

    return lambda_tau, results, fitted_values, intervector_distance_microns


# Load PIV data from PIVlab into dataframes
def load_piv_data(data_path, condition, subcondition, min_frame=0, max_frame=None, skip_frames=1):
    input_piv_data = os.path.join(data_path, condition, subcondition, "piv_data", "PIVlab_****.txt")
    
    # Using a for loop instead of list comprehension
    dfs = []
    for file in tqdm(sorted(glob.glob(input_piv_data))[min_frame:max_frame:skip_frames], desc=f"Loading PIV data for {condition} {subcondition}", leave=False):
        df = pd.read_csv(file, skiprows=2).fillna(0).rename(columns={
            "magnitude [m/s]": "velocity magnitude [m/s]",
            "simple shear [1/s]": "shear [1/s]",
            "simple strain [1/s]": "strain [1/s]",
            "Vector type [-]": "data type [-]"
        })
        dfs.append(df)

    return dfs

# Generate dataframes from PIV data with time intervals applied
def generate_dataframes_from_piv_data(data_path, condition, subcondition, min_frame=0, max_frame=None, skip_frames=1, plot_autocorrelation=True, time_interval=1):
    output_directory_dfs = os.path.join(data_path, condition, subcondition, "dataframes_PIV")
    os.makedirs(output_directory_dfs, exist_ok=True)

    # Load PIV data
    data_frames = load_piv_data(data_path, condition, subcondition, min_frame, max_frame, skip_frames)

    # Calculating mean values with valid vectors only
    mean_values = []
    for frame_id, data_frame in enumerate(tqdm(data_frames, desc=f"Generating dataframes for {condition} {subcondition}", leave=False)):
        lambda_tau, results, fitted_values, intervector_distance_microns = correlation_length(data_frame)
        if plot_autocorrelation:
            plot_autocorrelation_values(data_path, condition, subcondition, frame_id, lambda_tau * 1E6, results, fitted_values, intervector_distance_microns, time_interval)
        data_frame["correlation length [m]"] = lambda_tau

        data_frame = data_frame[data_frame["data type [-]"] == 1]
        mean_values.append(data_frame.mean(axis=0))

    # Creating mean DataFrame
    mean_data_frame = pd.DataFrame(mean_values)
    mean_data_frame.reset_index(drop=False, inplace=True)
    mean_data_frame.rename(columns={'index': 'frame'}, inplace=True)

    # Subtract the minimum row value for each column from the entire column for velocity magnitude
    mean_data_frame["velocity magnitude [m/s]"] = mean_data_frame["velocity magnitude [m/s]"] 
    
    # add a column with total distance travelled
    mean_data_frame["distance [m]"] = mean_data_frame["velocity magnitude [m/s]"].cumsum() * time_interval
    mean_data_frame["distance [m]"] = mean_data_frame["distance [m]"] - mean_data_frame["distance [m]"].min()

    # Calculate power and add to DataFrame
    volume = 2E-9  # µl --> m^3
    viscosity = 1E-3  # mPa*S

    # apply gaussian smoothing to correlation length
    mean_data_frame["correlation length [m]"] = gaussian_filter1d(mean_data_frame["correlation length [m]"], sigma=10, mode='nearest')

    mean_data_frame["power [W]"] = volume * viscosity * (mean_data_frame["velocity magnitude [m/s]"]/mean_data_frame["correlation length [m]"])**2

    # Calculate cumulative work
    mean_data_frame["work [J]"] = mean_data_frame["power [W]"].cumsum()

    # Scale time appropriately using the provided time_interval
    mean_data_frame["time (s)"] = mean_data_frame["frame"] * time_interval
    mean_data_frame["time (min)"] = mean_data_frame["time (s)"] / 60
    mean_data_frame["time (h)"] = mean_data_frame["time (min)"] / 60

    # Creating pivot matrices for each feature
    features = data_frames[0].columns[:-1]
    pivot_matrices = {feature: [] for feature in features}

    for data_frame in data_frames:
        temporary_dictionary = {feature: data_frame.pivot(index='y [m]', columns='x [m]', values=feature).values for feature in features}
        for feature in features:
            pivot_matrices[feature].append(temporary_dictionary[feature])

    pivot_data_frame = pd.DataFrame(pivot_matrices)

    # Adjusting column names in mean_data_frame
    mean_data_frame.columns = [f"{column}_mean" if column not in ["frame", "time (s)", "time (min)", "time (h)"] else column for column in mean_data_frame.columns]
    
    # Adding time column to pivot_data_frame
    pivot_data_frame["frame"] = mean_data_frame["frame"].values
    
    # subtract the minimum row value for each column from the entire column in 
    
    # Save DataFrames to CSV
    mean_df_output_path = os.path.join(output_directory_dfs, "mean_values.csv")
    mean_data_frame.to_csv(mean_df_output_path, index=False)

    pivot_df_output_path = os.path.join(output_directory_dfs, "features_matrices.csv")
    pivot_data_frame.to_csv(pivot_df_output_path, index=False)

    return mean_data_frame, pivot_data_frame



# Plot the PIVlab output as heatmaps
def generate_heatmaps_from_dataframes(df, data_path, condition, subcondition, feature_limits, time_interval=3):
    for feature, limits in feature_limits.items():
        vmin, vmax = limits

        for j in tqdm(range(len(df)), desc=f"Generating heatmaps for {condition} {subcondition} {feature}", leave=False):
            vals = df.iloc[j, df.columns.get_loc(feature)]

            output_directory_heatmaps = os.path.join(data_path, condition, subcondition, "heatmaps_PIV", f"{feature.split()[0]}", f"{feature.split()[0]}_heatmap_{j}.jpg")
            image_files_pattern = f"{data_path}/{condition}/{subcondition}/piv_movie_converted/converted_image_****.tif"
            image_files = sorted(glob.glob(image_files_pattern))[j]
            image = Image.open(image_files)

            plt.figure(figsize=(10, 6))
            plt.imshow(image, cmap=None, extent=[-2762/2, 2762/2, -2762/2, 2762/2]) # piv image
            im = plt.imshow(vals, cmap='inferno', origin='upper', alpha=0.7, extent=[-2762/2, 2762/2, -2762/2, 2762/2], vmin=vmin, vmax=vmax) # heatmap
            plt.xlabel('x [um]')
            plt.ylabel('y [um]')
            cbar = plt.colorbar(im)
            cbar.set_label(feature)
            time = df.iloc[j, -1]
            plt.title(f"PIV - {feature}  ||  time: {int(time * time_interval/60)} min -- {int(time * time_interval/3600)} hours")

            os.makedirs(os.path.dirname(output_directory_heatmaps), exist_ok=True)
            plt.savefig(output_directory_heatmaps, format='jpg', dpi=200)
            plt.close()


def create_movies_PIV(data_path, condition, subcondition, frame_rate, feature_limits=None, max_frame=None):
    plots_dir = f"{data_path}/{condition}/{subcondition}/heatmaps_PIV/"
    for feature in feature_limits.keys():
        feature_name_for_file = feature.split()[0]
        heatmap_dir = os.path.join(data_path, condition, subcondition, "heatmaps_PIV", f"{feature.split()[0]}", f"{feature.split()[0]}_heatmap_****.jpg")
        image_files = natsorted(glob.glob(heatmap_dir))

        if not image_files:
            print(f"No images found for feature {feature_name_for_file}.")
            continue

        # Limit the number of files if max_frame is specified
        image_files = image_files[:max_frame] if max_frame is not None else image_files

        # Get the resolution of the first image (assuming all images are the same size)
        first_image = cv2.imread(image_files[0])
        video_resolution = (first_image.shape[1], first_image.shape[0])  # Width x Height

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = f'{plots_dir}{feature_name_for_file}.avi'
        out = cv2.VideoWriter(out_path, fourcc, frame_rate, video_resolution)

        for file in tqdm(image_files, desc=f"Creating movie for {condition} {subcondition} {feature}", leave=False):
            img = cv2.imread(file)
            out.write(img)  # Write the image as is, without resizing

        out.release()
        print(f"Video saved to {out_path}")


# Process PIV data for all conditions and subconditions, then average and save results
def process_piv_data(data_path, conditions, subconditions, feature_limits, time_intervals, skip_frames, min_frame=0, max_frame=None, plot_autocorrelation=True, frame_rate=120, heatmaps=True):
    
    # exclude 'negative' from the conditions list
    conditions = [condition for condition in conditions if condition != 'negative']

    for i, condition in tqdm(enumerate(conditions), desc="Processing PIV data", total=len(conditions), leave=True):
        time_interval = time_intervals[i] * skip_frames
        results = []
        for subcondition in tqdm(subconditions, desc=f"Processing subconditions for {condition}", leave=False):
            m, p = generate_dataframes_from_piv_data(data_path, condition, subcondition, min_frame, max_frame, skip_frames, plot_autocorrelation, time_interval)
            results.append(m)

            if heatmaps == True:
                convert_images(data_path, conditions, subconditions, max_frame=None, brightness_factor=1, contrast_factor=1, skip_frames=skip_frames)
                generate_heatmaps_from_dataframes(p, data_path, condition, subcondition, feature_limits, time_interval)
                create_movies_PIV(data_path, condition, subcondition, frame_rate, feature_limits=feature_limits, max_frame=max_frame)

        # Averaging and saving the results for the current condition
        save_path = os.path.join(data_path, condition, 'averaged')
        average_df = sum(results) / len(results)
        
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        average_df.to_csv(os.path.join(save_path, f"{condition}_average.csv"))
        
        
######################################### PIV post-processing #########################################


def plot_PIV_features(data_path, conditions, subconditions, features_pca, sigma=10, min_frame=0, max_frame=None):
    for condition in tqdm(conditions, desc="Plotting PIV features", leave=True):
        dfs = []
        
        for subcondition in subconditions:
            # Construct the file path
            file_path = os.path.join(data_path, condition, subcondition, "dataframes_PIV", "mean_values.csv")
            df = pd.read_csv(file_path)

            # Apply Gaussian filter
            df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(lambda x: gaussian_filter1d(x, sigma=sigma))

            # Rename columns
            df = df.rename(columns={
                "data type [-]_mean": "work [J]",
                "correlation length [m]_mean": "correlation length [um]",
                "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]",
            })

            # Calculate cumulative work
            df["work [J]"] = df["power [W]_mean"].cumsum()

            # Slice the dataframe if min_frame and max_frame are provided
            df = df.iloc[min_frame:max_frame, :]

            dfs.append(df)

        # Plot individual features
        for feature in dfs[0].columns[5:-3]:
            plt.figure(figsize=(12, 8))
            for df, subcondition in zip(dfs, subconditions):
                output_directory_plots = os.path.join(data_path, "output_data", "PIV_plots", condition, subcondition)
                
                # Ensure the directory exists
                os.makedirs(output_directory_plots, exist_ok=True)
                
                plt.plot(df["time (h)"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition}_{subcondition}')
                plt.xlabel('Time (hours)')
                plt.ylabel(feature)
                plt.title(f"PIV - {feature}")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.savefig(os.path.join(output_directory_plots, f"{feature.split()[0]}_h.jpg"), format='jpg', dpi=200)
                plt.close()

                plt.plot(df["time (min)"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition}_{subcondition}')
                plt.xlabel('Time (minutes)')
                plt.ylabel(feature)
                plt.title(f"PIV - {feature}")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.savefig(os.path.join(output_directory_plots, f"{feature.split()[0]}_min.jpg"), format='jpg', dpi=200)
                plt.close()


def plot_PIV_features_averaged(data_path, conditions, features_pca, sigma=2, min_frame=0, max_frame=None):
    for condition in tqdm(conditions, desc="Plotting averaged PIV features", leave=True):
        averaged_data_path = os.path.join(data_path, condition, "averaged")
        averaged_df_file = os.path.join(averaged_data_path, f"{condition}_average.csv")
        
        if not os.path.exists(averaged_df_file):
            print(f"Error: Averaged dataframe {averaged_df_file} does not exist.")
            continue
        
        df = pd.read_csv(averaged_df_file)
        df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        df = df.rename(columns={"data type [-]_mean": "work [J]", "correlation length [m]_mean": "correlation length [um]", "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
        df["work [J]"] = df["power [W]_mean"].cumsum()
        df = df.iloc[min_frame:max_frame, :]
        
        output_directory_plots = os.path.join(data_path, "output_data", "PIV_plots_averaged", condition)
        os.makedirs(output_directory_plots, exist_ok=True)
        
        for feature in features_pca:
            plt.figure(figsize=(12, 8))
            plt.plot(df["time (h)"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1)
            plt.xlabel('Time (hours)')
            plt.ylabel(feature)
            plt.title(f"Averaged PIV - {feature} over Subconditions")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(output_directory_plots, f"{feature.split()[0]}_h.jpg"), format='jpg', dpi=200)
            plt.close()
            
            plt.plot(df["time (min)"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1)
            plt.xlabel('Time (minutes)')
            plt.ylabel(feature)
            plt.title(f"Averaged PIV - {feature} over Subconditions")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(os.path.join(output_directory_plots, f"{feature.split()[0]}_min.jpg"), format='jpg', dpi=200)
            plt.close()


def plot_PIV_features_combined(data_path, conditions, features_pca, sigma=2, min_frame=0, max_frame=None):
    combined_output_dir = os.path.join(data_path, "output_data", "PIV_plots", "averaged_conditions")
    os.makedirs(combined_output_dir, exist_ok=True)
    
    combined_data = {feature: {} for feature in features_pca}
    
    for condition in tqdm(conditions, desc="Collecting PIV data", leave=True):
        averaged_data_path = os.path.join(data_path, condition, "averaged")
        averaged_df_file = os.path.join(averaged_data_path, f"{condition}_average.csv")
        
        if not os.path.exists(averaged_df_file):
            print(f"Error: Averaged dataframe {averaged_df_file} does not exist.")
            continue
        
        df = pd.read_csv(averaged_df_file)
        df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(lambda x: gaussian_filter1d(x, sigma=sigma))
        df = df.rename(columns={"data type [-]_mean": "work [J]", "correlation length [m]_mean": "correlation length [um]", "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
        df["work [J]"] = df["power [W]_mean"].cumsum()
        df = df.iloc[min_frame:max_frame, :]
        
        for feature in features_pca:
            combined_data[feature][condition] = (df["time (h)"], df[feature])
    
    for feature in features_pca:
        plt.figure(figsize=(12, 8))
        
        for condition, (time, values) in combined_data[feature].items():
            plt.plot(time, values, marker='o', linestyle='-', markersize=1, linewidth=1, label=condition)
        
        plt.xlabel('Time (hours)')
        plt.ylabel(feature)
        plt.title(f"Combined PIV - {feature} across Conditions")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(os.path.join(combined_output_dir, f"combined_{feature.split()[0]}_h.jpg"), format='jpg', dpi=200)
        plt.close()

        plt.figure(figsize=(12, 8))
        
        for condition, (time, values) in combined_data[feature].items():
            time_in_min = time * 60
            plt.plot(time_in_min, values, marker='o', linestyle='-', markersize=1, linewidth=1, label=condition)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title(f"Combined PIV - {feature} across Conditions")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(os.path.join(combined_output_dir, f"combined_{feature.split()[0]}_min.jpg"), format='jpg', dpi=200)
        plt.close()


def plot_PIV_features_all_conditions_subconditions(data_path, conditions, subconditions, features_pca, sigma=2, min_frame=0, max_frame=None):
    """
    Generate PIV plots that display all conditions and subconditions together on the same plot for each feature.

    Args:
        data_path (str): Path to the data directory.
        conditions (list): List of conditions.
        subconditions (list): List of subconditions for each condition.
        features_pca (list): List of features to include in plotting.
        sigma (float): The standard deviation for Gaussian kernel applied for smoothing.
        min_frame (int): The minimum frame to include in the analysis.
        max_frame (int): The maximum frame to include in the analysis.
    """
    # Prepare output directory for combined plots in output_data/PIV_plots
    combined_output_dir = os.path.join(data_path, "output_data", "PIV_plots", "all_conditions_subconditions")
    os.makedirs(combined_output_dir, exist_ok=True)
    
    for feature in features_pca:
        plt.figure(figsize=(12, 8))
        
        for condition in conditions:
            for subcondition in subconditions:
                # Path to the subcondition data
                subcondition_data_path = os.path.join(data_path, condition, subcondition, "dataframes_PIV", "mean_values.csv")
                
                if not os.path.exists(subcondition_data_path):
                    print(f"Error: Data file {subcondition_data_path} does not exist.")
                    continue
                
                # Load the subcondition dataframe
                df = pd.read_csv(subcondition_data_path)
                
                # Apply Gaussian smoothing
                df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(lambda x: gaussian_filter1d(x, sigma=sigma))
                
                # Rename columns for consistency in plotting
                df = df.rename(columns={"data type [-]_mean": "work [J]", 
                                        "correlation length [m]_mean": "correlation length [um]", 
                                        "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
                
                df["work [J]"] = df["power [W]_mean"].cumsum()
                
                # Limit the frames if specified
                df = df.iloc[min_frame:max_frame, :]
                
                # Plot each subcondition on the same figure
                plt.plot(df["time (h)"], df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition} - {subcondition}')
        
        plt.xlabel('Time (hours)')
        plt.ylabel(feature)
        plt.title(f"All Conditions and Subconditions Combined - {feature}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='best', fontsize='small', ncol=2)  # Adjust legend to fit all entries
        plt.savefig(os.path.join(combined_output_dir, f"combined_{feature.split()[0]}_h.jpg"), format='jpg', dpi=200)
        plt.close()

        plt.figure(figsize=(12, 8))
        
        for condition in conditions:
            for subcondition in subconditions:
                # Load the subcondition dataframe again
                df = pd.read_csv(os.path.join(data_path, condition, subcondition, "dataframes_PIV", "mean_values.csv"))
                df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(lambda x: gaussian_filter1d(x, sigma=sigma))
                df = df.rename(columns={"data type [-]_mean": "work [J]", 
                                        "correlation length [m]_mean": "correlation length [um]", 
                                        "velocity magnitude [m/s]_mean": "velocity magnitude [um/s]"})
                df["work [J]"] = df["power [W]_mean"].cumsum()
                df = df.iloc[min_frame:max_frame, :]
                
                time_in_min = df["time (h)"] * 60  # Convert hours to minutes for the second plot
                plt.plot(time_in_min, df[feature], marker='o', linestyle='-', markersize=1, linewidth=1, label=f'{condition} - {subcondition}')
        
        plt.xlabel('Time (minutes)')
        plt.ylabel(feature)
        plt.title(f"All Conditions and Subconditions Combined - {feature}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='best', fontsize='small', ncol=2)  # Adjust legend to fit all entries
        plt.savefig(os.path.join(combined_output_dir, f"combined_{feature.split()[0]}_min.jpg"), format='jpg', dpi=200)
        plt.close()



def plot_PIV_all(data_path, conditions, subconditions, features_pca, min_frame=0, max_frame=None):
    """
    This function processes PIV data, applies Gaussian smoothing to correlation lengths, 
    and generates PCA and feature plots for all conditions and subconditions, while removing 
    the 'negative' condition.

    Args:
        data_path (str): The path to the PIV data.
        conditions (list): A list of conditions to process.
        subconditions (list): A list of subconditions to process.
        features_pca (list): A list of features to use for PCA plotting.
        min_frame (int): The minimum frame to process (default=0).
        max_frame (int): The maximum frame to process (default=None).
        smoothing_sigma (float): The sigma value for Gaussian smoothing (default=2).
        time_intervals (list): List of time intervals for each condition (default=None).
        skip_frames (int): Frame skipping interval (default=1).
    """
    # Remove 'negative' from conditions list
    conditions = [cond for cond in conditions if cond.lower() != 'negative']

    
    # Process and plot PIV features
    plot_PIV_features(
        data_path, 
        conditions,
        subconditions, 
        features_pca, 
        min_frame=min_frame, 
        max_frame=max_frame
    )

    # Process and plot averaged PIV features
    plot_PIV_features_averaged(
        data_path, 
        conditions,
        features_pca, 
        min_frame=min_frame, 
        max_frame=max_frame
    )

    # Process and plot combined PIV features
    plot_PIV_features_combined(
        data_path, 
        conditions,
        features_pca, 
        min_frame=min_frame, 
        max_frame=max_frame
    )

    # Process and plot all conditions and subconditions
    plot_PIV_features_all_conditions_subconditions(
        data_path, 
        conditions,
        subconditions, 
        features_pca, 
        min_frame=min_frame, 
        max_frame=max_frame
    )


def combine_averaged_dataframes(data_path, conditions, subconditions, output_file_name="combined_PIV.csv"):
    all_dataframes = []
    
    for condition in tqdm(conditions, desc="Conditions"):
        for subcondition in tqdm(subconditions, desc=f"Subconditions for {condition}", leave=False):
            # Define the path to the 'averaged' folder for each condition and subcondition
            averaged_dir = os.path.join(data_path, condition, 'averaged')
            
            # Check if the 'averaged' directory exists and has CSV files
            if os.path.exists(averaged_dir):
                # Load the averaged CSV file
                averaged_csv = os.path.join(averaged_dir, f"{condition}_average.csv")
                
                if os.path.isfile(averaged_csv):
                    # Read the CSV into a DataFrame
                    df = pd.read_csv(averaged_csv)
                    
                    # Add 'condition' and 'subcondition' columns to track the origin of the data
                    df['condition'] = condition
                    df['subcondition'] = subcondition
                    
                    # Append the DataFrame to the list
                    all_dataframes.append(df)
                else:
                    print(f"No averaged CSV found for {condition}/{subcondition}. Skipping...")
            else:
                print(f"'Averaged' folder not found for {condition}/{subcondition}. Skipping...")
    
    # Concatenate all DataFrames into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
    else:
        combined_df = pd.DataFrame()  # Return an empty DataFrame if nothing was found

    # Save the combined DataFrame to the "output_data" directory of each condition
    output_dir = os.path.join(data_path, "output_data")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Save the combined DataFrame as a CSV file in the "output_data" folder
    output_file_path = os.path.join(output_dir, output_file_name)
    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined DataFrame saved to: {output_file_path}")






######################################### Expression + PIV #########################################



def merge_expression_piv_data(data_path, output_folder="output_data", expression_file="combined_expression.csv", piv_file="combined_PIV.csv", output_file_name="merged_expression_PIV.csv"):
    # Define the full paths to the CSV files
    expression_file_path = os.path.join(data_path, output_folder, expression_file)
    piv_file_path = os.path.join(data_path, output_folder, piv_file)
    
    # Load the expression and PIV DataFrames
    expression_df = pd.read_csv(expression_file_path)
    piv_df = pd.read_csv(piv_file_path)
    
    # Standardize the column names to match
    expression_df.rename(columns={'Condition': 'condition', 'Subcondition': 'subcondition'}, inplace=True)

    # Determine which dataframe is longer and set the merge type accordingly
    merge_how = 'left' if len(expression_df) > len(piv_df) else 'right'

    # Merge the two DataFrames on the common columns: 'time (s)', 'condition', and 'subcondition'
    merged_df = pd.merge(expression_df, piv_df, on=['time (s)', 'condition', 'subcondition'], how='outer')
    
    # Define the output directory path
    output_dir = os.path.join(data_path, output_folder)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    # Save the merged DataFrame in the "output_data" directory
    output_file_path = os.path.join(output_dir, output_file_name)
    merged_df.to_csv(output_file_path, index=False)
    
    print(f"Merged DataFrame saved to: {output_file_path}")



def sanitize_filename(name):
    """Helper function to replace spaces and special characters in filenames."""
    return name.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_expression_piv(data_path, conditions, x_column, y_column, output_folder="output_data", 
                        merged_file="merged_expression_PIV.csv", plot_output_folder="output_data/expression_piv_plots", 
                        sigma_x=None, sigma_y=1, x_log=False, y_log=False, min_frame=0, max_frame=None, 
                        individual_plots=True, fill_na_method=None):
    """
    Plots the specified x_column vs y_column from the DataFrame for each condition and also generates a combined plot.
    
    Parameters:
    - data_path: Path to the data folder.
    - conditions: List of conditions to plot.
    - x_column: The column to use for the x-axis.
    - y_column: The column to use for the y-axis (with optional Gaussian smoothing).
    - output_folder: Folder where the merged data is stored.
    - merged_file: The merged CSV file name.
    - plot_output_folder: Folder where plots will be saved.
    - sigma_x: Gaussian smoothing factor to apply to the x-axis data (if None, no smoothing applied).
    - sigma_y: Gaussian smoothing factor to apply to the y-axis data (if None, no smoothing applied).
    - x_log: If True, set x-axis to log scale. Default is False.
    - y_log: If True, set y-axis to log scale. Default is False.
    - min_frame: Minimum frame to slice the DataFrame. Default is 0.
    - max_frame: Maximum frame to slice the DataFrame. If None, all frames after min_frame are used.
    - individual_plots: If True, generate individual plots for each condition. Default is True.
    - fill_na_method: Method to fill NA values ('ffill', 'bfill', 'zero', None). If None, NA values are dropped.
    """
    
    # Load the merged DataFrame from the output_data folder
    merged_file_path = os.path.join(data_path, output_folder, merged_file)
    merged_df = pd.read_csv(merged_file_path)

    # Handle NA values based on the fill_na_method
    if fill_na_method == 'ffill':
        merged_df.fillna(method='ffill', inplace=True)
    elif fill_na_method == 'bfill':
        merged_df.fillna(method='bfill', inplace=True)
    elif fill_na_method == 'zero':
        merged_df.fillna(0, inplace=True)
    else:
        merged_df.dropna(subset=[x_column, y_column], inplace=True)  # Drop rows where x or y are NaN

    # Slice the DataFrame based on the min_frame and max_frame for each condition
    if min_frame > 0 or max_frame is not None:
        sliced_dfs = []
        for condition in conditions:
            condition_df = merged_df[merged_df['condition'] == condition]
            if max_frame is not None:
                condition_df = condition_df.iloc[min_frame:max_frame]
            else:
                condition_df = condition_df.iloc[min_frame:]
            sliced_dfs.append(condition_df)
        merged_df = pd.concat(sliced_dfs)

    # Define the output folder for plots
    plot_output_dir = os.path.join(data_path, plot_output_folder)
    os.makedirs(plot_output_dir, exist_ok=True)  # Ensure the output directory exists

    # Initialize a combined plot for all conditions
    plt.figure(figsize=(10, 8))
    
    # Loop through each condition and plot the specified columns
    for condition in conditions:
        # Filter the DataFrame for the current condition
        condition_df = merged_df[merged_df['condition'] == condition]
        
        if condition_df.empty:
            print(f"No data available for condition: {condition}")
            continue
        
        # Apply Gaussian smoothing to the selected x-axis and y-axis columns
        smoothed_x = gaussian_filter1d(condition_df[x_column], sigma=sigma_x) if sigma_x is not None else condition_df[x_column]
        smoothed_y = gaussian_filter1d(condition_df[y_column], sigma=sigma_y) if sigma_y is not None else condition_df[y_column]
        
        # Filter out non-positive values if log scale is applied to either axis
        if x_log and y_log:
            positive_mask = (smoothed_x > 0) & (smoothed_y > 0)  # Both x and y must be positive
        elif x_log:
            positive_mask = smoothed_x > 0  # Only x must be positive
        elif y_log:
            positive_mask = smoothed_y > 0  # Only y must be positive
        else:
            positive_mask = np.ones(len(smoothed_x), dtype=bool)  # No filtering if log is not applied
        
        # Apply the positive mask to filter both x and y
        smoothed_x = smoothed_x[positive_mask]
        smoothed_y = smoothed_y[positive_mask]
        
        # Sanitize the column names and condition for use in filenames
        sanitized_x_column = sanitize_filename(x_column)
        sanitized_y_column = sanitize_filename(y_column)
        sanitized_condition = sanitize_filename(condition)
        
        # Generate individual plots if individual_plots is True
        if individual_plots:
            plt.figure(figsize=(8, 6))
            plt.plot(smoothed_x, smoothed_y, label=condition)
            plt.title(f'{x_column} vs {y_column} for Condition: {condition}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
            # Set log scale if specified
            if x_log:
                plt.xscale('log')
            if y_log:
                plt.yscale('log')
            
            plt.grid(True)

            # Save the individual plot
            output_file = os.path.join(plot_output_dir, f"{sanitized_x_column}_vs_{sanitized_y_column}_{sanitized_condition}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Plot saved for condition {condition} at {output_file}")

        # Add the condition's plot to the combined figure
        plt.plot(smoothed_x, smoothed_y, label=condition)

    # Finalize the combined plot with all conditions
    plt.title(f'{x_column} vs {y_column} for All Conditions (Smoothed)')
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    # Set log scale if specified
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    
    plt.grid(True)
    plt.legend()
    
    # Save the combined plot
    combined_plot_file = os.path.join(plot_output_dir, f"{sanitized_x_column}_vs_{sanitized_y_column}_All_Conditions.png")
    plt.savefig(combined_plot_file, dpi=300)
    plt.close()
    
    print(f"Combined plot saved at {combined_plot_file}")

    return merged_df




def plot_pca_expression_piv(data_path, conditions, subconditions, features, sigma=1, merged_file="merged_expression_PIV.csv", output_folder="output_data/expression_piv_plots"):
    """
    Perform PCA on the specified features from the merged data and save the PCA plot with gradient lines connecting points.
    Additionally, generate individual PCA plots for each condition and subcondition.
    
    Parameters:
    - data_path: Path to the data folder.
    - conditions: List of conditions to include in the PCA.
    - subconditions: List of subconditions to include in the PCA.
    - features: List of feature columns to use in the PCA.
    - sigma: Standard deviation for Gaussian smoothing.
    - merged_file: The merged CSV file to read the data from.
    - output_folder: Folder where the PCA plot will be saved.
    """
    # Load the merged DataFrame from the output_data folder
    merged_file_path = os.path.join(data_path, "output_data", merged_file)
    merged_df = pd.read_csv(merged_file_path)
    
    # Get available columns in the DataFrame
    available_columns = merged_df.columns
    print("Available columns in the DataFrame:", available_columns)

    # Filter features to keep only those present in the DataFrame
    filtered_features = [feature for feature in features if feature in available_columns]
    if len(filtered_features) < len(features):
        missing_features = set(features) - set(filtered_features)
        print(f"Warning: The following features are missing and will be excluded: {missing_features}")
    
    # Apply Gaussian smoothing to the filtered features
    merged_df[filtered_features] = merged_df[filtered_features].apply(lambda x: gaussian_filter1d(x, sigma=sigma))

    # Combine all conditions and subconditions for PCA
    combined_df = pd.DataFrame()

    # Iterate over all combinations of conditions and subconditions
    for condition in conditions:
        for subcondition in subconditions:
            # Filter the DataFrame for the current condition and subcondition
            condition_df = merged_df[(merged_df['condition'] == condition) & (merged_df['subcondition'] == subcondition)]
            
            # Drop rows where any of the features have NaN values
            condition_df = condition_df.dropna(subset=filtered_features)

            if condition_df.shape[0] < 1:
                print(f"Warning: Not enough valid samples for {condition} and {subcondition}. Skipping...")
                continue
            
            condition_df["combined"] = f"{condition}_{subcondition}"  # Add a combined label
            combined_df = pd.concat([combined_df, condition_df])

    if combined_df.empty:
        print("Error: No data found for the specified conditions and subconditions.")
        return

    # Perform PCA on the combined data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(combined_df.loc[:, filtered_features])
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf["combined"] = combined_df["combined"].reset_index(drop=True)

    # Initialize the plot for all conditions combined
    plt.figure(figsize=(10, 6))

    # Set up color palette for plotting
    sns.set_palette("colorblind", color_codes=True)
    unique_combinations = principalDf["combined"].unique()
    colors = sns.color_palette("colorblind", n_colors=len(unique_combinations))

    # Plot all conditions on the same plot with gradient lines connecting the dots
    for i, combined_label in enumerate(unique_combinations):
        subset = principalDf[principalDf["combined"] == combined_label]
        num_points = subset.shape[0]
        alphas = np.linspace(0.01, 1, num_points)  # Alpha values linearly spaced from 0.01 to 1
        
        # Plot gradient lines connecting the points
        for j in range(1, num_points):
            plt.plot(subset['principal component 1'][j-1:j+1], subset['principal component 2'][j-1:j+1], 
                     alpha=alphas[j], linestyle='-', linewidth=2, color=colors[i % len(colors)])

        # Plot the points
        plt.scatter(subset['principal component 1'], subset['principal component 2'], 
                    alpha=0.5, label=combined_label, s=10, color=colors[i % len(colors)])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Selected Features (All Conditions and Subconditions)')
    plt.legend()
    plt.grid(True)

    # Save the combined PCA plot in the output folder
    output_dir_pca = os.path.join(data_path, output_folder)
    os.makedirs(output_dir_pca, exist_ok=True)
    pca_output_file = os.path.join(output_dir_pca, "PCA_plot_all_conditions_gradient.jpg")
    plt.savefig(pca_output_file, format='jpg', dpi=200)
    plt.close()

    print(f"PCA plot with gradient lines saved at {pca_output_file}")

    # Now generate individual PCA plots for each condition and subcondition
    for condition in conditions:
        for subcondition in subconditions:
            # Filter the DataFrame for the current condition and subcondition
            condition_df = merged_df[(merged_df['condition'] == condition) & (merged_df['subcondition'] == subcondition)]
            
            # Drop rows where any of the features have NaN values
            condition_df = condition_df.dropna(subset=filtered_features)

            if condition_df.shape[0] < 1:
                print(f"Warning: Not enough valid samples for {condition} and {subcondition}. Skipping...")
                continue

            # Perform PCA on the current subset
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(condition_df.loc[:, filtered_features])
            principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

            # Initialize plot for the current condition and subcondition
            plt.figure(figsize=(10, 6))

            # Plot with gradient lines for individual conditions
            num_points = principalDf.shape[0]
            alphas = np.linspace(0.01, 1, num_points)
            for j in range(1, num_points):
                plt.plot(principalDf['principal component 1'][j-1:j+1], principalDf['principal component 2'][j-1:j+1], 
                         alpha=alphas[j], linestyle='-', linewidth=2, color='b')

            # Plot the points
            plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], 
                        alpha=0.5, label=f'{condition}_{subcondition}', s=10, color='b')

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA of {condition}_{subcondition}')
            plt.grid(True)

            # Save the individual PCA plot
            pca_output_file = os.path.join(output_dir_pca, f"PCA_plot_{condition}_{subcondition}_gradient.jpg")
            plt.savefig(pca_output_file, format='jpg', dpi=200)
            plt.close()

            print(f"Individual PCA plot for {condition}_{subcondition} saved at {pca_output_file}")



def delete_outputs(data_path, conditions, subconditions, output_dirs=None):
    """
    Deletes all output files and directories for the given conditions and subconditions,
    and removes everything inside 'output_data' except the 'movies' directory.

    Args:
        data_path (str): Base directory for PIV data and output.
        conditions (list): List of conditions.
        subconditions (list): List of subconditions.
        output_dirs (list, optional): Specific output directories to delete. If None, delete all known output directories.
    """
    # Default output directories to remove
    if output_dirs is None:
        output_dirs = [
            "piv_movie_converted",
            "autocorrelation_plots",
            "dataframes_PIV",
            "heatmaps_PIV",
            "plots_PIV",
            "averaged",
            os.path.join("PIV_plots", "averaged_conditions"),
            os.path.join("PIV_plots", "all_conditions_subconditions"),
            "combined_expression_PIV_plots",
            "expression_expression_piv_plots"  # Add expression_piv_plots to the list
        ]

    for condition in conditions:
        for subcondition in subconditions:
            for output_dir in output_dirs:
                dir_path = os.path.join(data_path, condition, subcondition, output_dir)
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted directory: {dir_path}")
                    except Exception as e:
                        print(f"Error deleting directory {dir_path}: {e}")

        # Remove the averaged directory at the condition level
        averaged_dir = os.path.join(data_path, condition, "averaged")
        if os.path.exists(averaged_dir):
            try:
                shutil.rmtree(averaged_dir)
                print(f"Deleted directory: {averaged_dir}")
            except Exception as e:
                print(f"Error deleting directory {averaged_dir}: {e}")

    # Remove the combined plots directories at the top level
    combined_dirs = [
        "combined_PIV_plots",
        os.path.join("PIV_plots", "averaged_conditions"),
        os.path.join("PIV_plots", "all_conditions_subconditions")
    ]

    for combined_dir in combined_dirs:
        combined_dir_path = os.path.join(data_path, combined_dir)
        if os.path.exists(combined_dir_path):
            try:
                shutil.rmtree(combined_dir_path)
                print(f"Deleted directory: {combined_dir_path}")
            except Exception as e:
                print(f"Error deleting directory {combined_dir_path}: {e}")

    # Remove the PIV_plots directory in the data_path
    piv_plots_dir = os.path.join(data_path, "PIV_plots")
    if os.path.exists(piv_plots_dir):
        try:
            shutil.rmtree(piv_plots_dir)
            print(f"Deleted directory: {piv_plots_dir}")
        except Exception as e:
            print(f"Error deleting directory {piv_plots_dir}: {e}")
    
    # Remove the expression_piv_plots directory in the data_path
    expression_piv_plots_dir = os.path.join(data_path, "output_data", "expression_piv_plots")
    if os.path.exists(expression_piv_plots_dir):
        try:
            shutil.rmtree(expression_piv_plots_dir)
            print(f"Deleted directory: {expression_piv_plots_dir}")
        except Exception as e:
            print(f"Error deleting directory {expression_piv_plots_dir}: {e}")
    
    # Remove all folders in output_data except 'movies'
    output_data_dir = os.path.join(data_path, "output_data")
    if os.path.exists(output_data_dir):
        for item in os.listdir(output_data_dir):
            item_path = os.path.join(output_data_dir, item)
            if os.path.isdir(item_path) and item != 'movies':
                try:
                    shutil.rmtree(item_path)
                    print(f"Deleted directory in output_data: {item_path}")
                except Exception as e:
                    print(f"Error deleting directory {item_path}: {e}")
            elif os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    print(f"Deleted file in output_data: {item_path}")
                except Exception as e:
                    print(f"Error deleting file {item_path}: {e}")