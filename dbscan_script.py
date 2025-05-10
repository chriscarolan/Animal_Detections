import sys
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import logging

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder # Needed for N03 Colab pipeline
from sklearn.decomposition import PCA # Needed for N03 Colab pipeline

# Attempt to import TensorFlow and Keras components
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array # Not used in snippet, but good to have
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("PYTHON_SCRIPT_WARNING: TensorFlow/Keras not found. N03 photo processing will be skipped.")

# --- Global Configuration & Model Loading ---
IMAGE_FILENAME_COLUMN_IN_CSV = 'filename' # IMPORTANT: Set this to the name of the column in your CSV that contains image filenames for N03 data
N03_IMAGE_SIZE = (224, 224)
N03_PCA_COMPONENTS = 8
N03_DBSCAN_EPS = 10
N03_DBSCAN_MIN_SAMPLES = 1

BASE_RESNET_MODEL = None
if TF_AVAILABLE:
    try:
        BASE_RESNET_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("PYTHON_SCRIPT: ResNet50 base model loaded successfully.")
    except Exception as e:
        print(f"PYTHON_SCRIPT_ERROR: Could not load ResNet50 model: {e}")
        TF_AVAILABLE = False # Disable TF processing if model load fails

def preprocess_single_image_for_resnet(image_path, target_size):
    """Loads, resizes, and converts a single image to a NumPy array for ResNet."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        numpydata = np.asarray(image)
        return numpydata
    except Exception as e:
        print(f"PYTHON_SCRIPT_ERROR: Skipping image {os.path.basename(image_path)} due to error: {e}")
        return None

# This function is a placeholder from previous versions,
# your Colab logic is now integrated directly into main for N03.
# We keep it so the script doesn't break if called with old assumptions, but it won't be used by N03.
def extract_features_from_image(image_path):
    print(f"PYTHON_SCRIPT: (Legacy Placeholder) Extracting features from: {image_path}")
    return np.random.rand(10).tolist()


def run_fallback_dbscan(df_subset, eps_val, min_samples_val):
    """Runs DBSCAN on Latitude/Longitude for a subset of the DataFrame."""
    print(f"PYTHON_SCRIPT: Running fallback DBSCAN (Lat/Lon) with eps={eps_val}, min_samples={min_samples_val}")
    if df_subset.empty:
        print("PYTHON_SCRIPT: Empty DataFrame for fallback DBSCAN, skipping.")
        return pd.Series(index=df_subset.index, dtype='int')

    fallback_cols = ['Latitude', 'Longitude']
    if not all(col in df_subset.columns for col in fallback_cols):
        print(f"PYTHON_SCRIPT_ERROR: Fallback columns {fallback_cols} not found. Skipping fallback DBSCAN.")
        return pd.Series([-1] * len(df_subset), index=df_subset.index, dtype='int')

    data_for_fallback_dbscan = df_subset[fallback_cols].copy()
    data_for_fallback_dbscan.fillna(0.0, inplace=True) # Handle NaNs

    # Note: As per user's last script version, fallback does not scale these features.
    # If scaling is desired for fallback, uncomment StandardScaler lines.
    # scaler_fallback = StandardScaler()
    # data_scaled_fallback = scaler_fallback.fit_transform(data_for_fallback_dbscan.values)
    # dbscan_fallback = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    # clusters = dbscan_fallback.fit_predict(data_scaled_fallback)
    
    dbscan_fallback = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    clusters = dbscan_fallback.fit_predict(data_for_fallback_dbscan.values)
    
    print(f"PYTHON_SCRIPT: Fallback DBSCAN complete. Cluster assignments: {pd.Series(clusters).value_counts().to_dict()}")
    return pd.Series(clusters, index=df_subset.index)


def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "PYTHON_SCRIPT_ERROR: Insufficient arguments. Expected input_csv, output_csv, eps, photo_paths_json."}))
        sys.exit(1)

    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2] # This is the main output CSV for the combined df_main
    cli_eps_value_str = sys.argv[3]
    photo_location_paths_json_str = sys.argv[4]

    # Initialize process_n03_with_photos and photo_location_paths_map early for summary
    process_n03_with_photos = False # Default
    photo_location_paths_map = {}
    try:
        photo_location_paths_map = json.loads(photo_location_paths_json_str)
        if photo_location_paths_map.get("N03"):
            process_n03_with_photos = True
    except json.JSONDecodeError:
        # If JSON is invalid, photo_location_paths_map remains empty, process_n03_with_photos remains False
        print_debug("Warning: Could not parse photo_location_paths_json. Proceeding without photo-specific processing.")
        pass # Keep default values


    try:
        cli_eps_value = float(cli_eps_value_str)
    except ValueError:
        print_debug(f"Warning - ValueError parsing EPS from '{cli_eps_value_str}'. Using default EPS: {DEFAULT_DBSCAN_EPS}")
        cli_eps_value = DEFAULT_DBSCAN_EPS
    print(f"PYTHON SCRIPT: Using FINAL EPS value: {cli_eps_value} for processing.", file=sys.stderr)
    print_debug(f"Received photo_paths: {photo_location_paths_json_str}")


    all_dataframes = []
    all_groups_summary = {}
    expected_groups = ['N01', 'N02', 'N03', 'T01', 'T02', 'T03']

    try:
        main_df_reader = pd.read_csv(input_csv_path, chunksize=100000, low_memory=False)
        temp_df_list = []
        for chunk in main_df_reader:
            temp_df_list.append(chunk)
        df_input_full = pd.concat(temp_df_list, ignore_index=True)
        
        if 'group_id' not in df_input_full.columns:
            print(json.dumps({"error": "PYTHON_SCRIPT_ERROR: 'group_id' column missing from input CSV."}))
            sys.exit(1)

    except FileNotFoundError:
        print(json.dumps({"error": f"PYTHON_SCRIPT_ERROR: Input CSV file not found at {input_csv_path}"}))
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(json.dumps({"error": f"PYTHON_SCRIPT_ERROR: Input CSV file is empty at {input_csv_path}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"PYTHON_SCRIPT_ERROR: Error reading CSV: {e}\n"}))
        sys.exit(1)

    for group_name in expected_groups:
        group_df = df_input_full[df_input_full['group_id'] == group_name].copy()
        
        if group_df.empty:
            print_debug(f"No data for group {group_name}. Skipping.")
            all_groups_summary[group_name] = {
                "total_clusters": 0,
                "inconsistent_clusters_count": 0,
                "total_animal_records": 0,
                "inconsistent_details": [],
                "used_image_model": False,
                "message": "No data found for this group."
            }
            # Add an empty 'cluster' column if it's expected by pd.concat later
            group_df['cluster'] = pd.NA 
            all_dataframes.append(group_df)
            continue

        photo_folder_path = photo_location_paths_map.get(group_name)
        
        # Determine EPS for the current group
        current_eps = N03_DBSCAN_EPS if group_name == 'N03' and process_n03_with_photos else cli_eps_value

        print_debug(f"Processing group: {group_name} with EPS: {current_eps}")
        
        processed_df, group_summary = process_group_data(
            group_df, 
            group_name, 
            current_eps, 
            photo_folder_path,
            (group_name == 'N03' and process_n03_with_photos) # Pass actual model usage flag
        )
        all_dataframes.append(processed_df)
        all_groups_summary[group_name] = group_summary

    if not all_dataframes:
        # This case should ideally be handled if df_input_full was empty or no groups had data
        print(json.dumps({"error": "PYTHON_SCRIPT_ERROR: No dataframes were processed to combine."}))
        sys.exit(1)
        
    df_main = pd.concat(all_dataframes, ignore_index=True)
    df_main.to_csv(output_csv_path, index=False)
    print_debug(f"Main clustered CSV saved to: {output_csv_path}")

    # --- Prepare the final JSON output structure for server.js ---
    final_json_output = {
        "output_file": os.path.basename(output_csv_path), # Basename of the main output CSV
        "clusters": { # This is the 'clusters' object server.js expects
            "message": "Clustering process completed for all groups.",
            "summary": {
                "eps_used_fallback": cli_eps_value, 
                "eps_used_n03_photos": N03_DBSCAN_EPS if process_n03_with_photos and ('N03' in photo_location_paths_map) else "N/A",
                "total_records_processed": len(df_input_full) if 'df_input_full' in locals() else "N/A",
                # You can add more overall summary fields here if needed,
                # e.g., total distinct clusters across all groups (requires iterating df_main['cluster'])
            },
            "details_by_group": all_groups_summary # Nest the per-group details here
        }
    }

    try:
        print(json.dumps(final_json_output))
    except Exception as e:
        # Fallback JSON if the main one fails to serialize
        error_output = {
            "output_file": os.path.basename(output_csv_path) if 'output_csv_path' in locals() else "unknown.csv",
            "clusters": {
                "message": f"Error during final JSON generation: {str(e)}",
                "summary": {"error_details": str(e)},
                "details_by_group": {} # Empty details on error
            },
            "error": "Failed to generate full JSON output from Python script."
        }
        print(json.dumps(error_output))
        # Also print to stderr for server.js to catch if stdout is compromised
        print(f"PYTHON_SCRIPT_ERROR: Could not serialize final JSON output: {e}", file=sys.stderr)

if __name__ == '__main__':
    main() 