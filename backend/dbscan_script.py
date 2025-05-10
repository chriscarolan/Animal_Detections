import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sys
import json
import re
import os
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

# Define print_debug function
def print_debug(message):
    """Prints a debug message to stderr."""
    print(f"DEBUG PY (AddBack Phase): {message}", file=sys.stderr)

# --- Conditionally import image processing libraries ---
PIL_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print_debug("Pillow (PIL) loaded.")
except ImportError:
    print_debug("Pillow (PIL) not found. Image processing will be disabled.")

try:
    import tensorflow as tf
    # Check TensorFlow version for Keras path
    tf_version = tuple(map(int, tf.__version__.split('.')))
    if tf_version >= (2, 0, 0):
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import Model
    else: # Fallback for older TensorFlow 1.x if necessary, though less likely now
        from keras.applications.resnet50 import ResNet50, preprocess_input
        from keras.preprocessing.image import img_to_array
        from keras.models import Model
    TENSORFLOW_AVAILABLE = True
    print_debug(f"TensorFlow and Keras components loaded (TF version: {tf.__version__}).")
except ImportError:
    print_debug("TensorFlow or Keras not found. Image-based clustering will be disabled.")
except AttributeError: # Catch if tf.__version__ is not available for some reason
    print_debug("Could not determine TensorFlow version. Assuming Keras path might fail.")
    TENSORFLOW_AVAILABLE = False


BASE_RESNET_MODEL = None

def load_resnet_model():
    global BASE_RESNET_MODEL
    if not TENSORFLOW_AVAILABLE:
        print("DEBUG PY: Cannot load ResNet50 model, TensorFlow is not available.")
        return None
    if BASE_RESNET_MODEL is None:
        try:
            # Load ResNet50 model, excluding the top classification layer, using average pooling
            BASE_RESNET_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print("DEBUG PY: ResNet50 model loaded successfully.")
        except Exception as e:
            print(f"DEBUG PY: Error loading ResNet50 model: {e}")
            return None
    return BASE_RESNET_MODEL

def create_site_column(df):
    """Replicates the Site column creation from the R script."""
    id_col_for_site = 'deployment_id'
    if id_col_for_site not in df.columns:
        if 'photo_location' in df.columns and id_col_for_site == 'deployment_id':
            id_col_for_site = 'photo_location'
        elif 'deployment_id' in df.columns and id_col_for_site == 'photo_location':
             id_col_for_site = 'deployment_id'
        else:
            print(f"Warning: Column '{id_col_for_site}' not found for Site creation. Skipping Site column.")
            return df

    df['Site'] = df[id_col_for_site].astype(str).apply(lambda x: re.sub(r'.*_', '', x))
    return df

def run_dbscan_on_location(df_location_original, eps_seconds, group_key_val, min_samples_val=1):
    """
    Runs DBSCAN on the 'timestamp_seconds' for a specific location group.
    Handles timestamp conversion and NA dropping internally for the group.
    """
    print_debug(f"\n--- Running DBSCAN for Group: {group_key_val} with EPS_SECONDS: {eps_seconds}, MIN_SAMPLES: {min_samples_val} ---")
    
    # Make a copy to avoid SettingWithCopyWarning on the original slice
    df_location = df_location_original.copy()

    # --- Per-Group Timestamp Processing ---
    if 'timestamp' not in df_location.columns:
        print_debug(f"[{group_key_val}] 'timestamp' column missing. Cannot proceed with this group.")
        return pd.DataFrame(), 0, 0, 0, [], False # clusters, inconsistent, animals, details, used_image_model

    df_location['timestamp'] = pd.to_datetime(df_location['timestamp'], errors='coerce')
    records_in_group_before_dropna = len(df_location)
    df_location.dropna(subset=['timestamp'], inplace=True)
    records_in_group_after_dropna = len(df_location)
    
    print_debug(f"[{group_key_val}] Timestamp processing: {records_in_group_before_dropna} initial records, "
                f"{records_in_group_after_dropna} records after coercing and dropping NaT timestamps.")

    if records_in_group_after_dropna == 0:
        print_debug(f"[{group_key_val}] No valid timestamp records remaining. Skipping DBSCAN.")
        return pd.DataFrame(), 0, 0, 0, [], False 

    # Calculate timestamp_seconds relative to this group's minimum valid timestamp
    df_location['timestamp_seconds'] = (df_location['timestamp'] - df_location['timestamp'].min()).dt.total_seconds()
    # --- End Per-Group Timestamp Processing ---

    # Ensure 'common_name' column exists, fill NA if necessary for inconsistency check
    if 'common_name' not in df_location.columns:
        print_debug(f"[{group_key_val}] 'common_name' column missing. Adding empty 'common_name' column.")
        df_location['common_name'] = "Unknown" # Or pd.NA
    else:
        df_location['common_name'] = df_location['common_name'].fillna("Unknown")

    # Prepare features for DBSCAN (only timestamp_seconds)
    features = df_location[['timestamp_seconds']]
    
    # Scale features
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except ValueError as e:
        print_debug(f"[{group_key_val}] ValueError during scaling (e.g., only one sample): {e}. Skipping DBSCAN for this group.")
        return pd.DataFrame(), 0, 0, records_in_group_after_dropna, [], False


    # Adjust eps for scaled data: eps_seconds is the desired time window in original seconds
    # scaler.scale_[0] is the standard deviation of timestamp_seconds for this group
    # We want eps for DBSCAN to represent eps_seconds in the scaled space.
    if scaler.scale_[0] == 0: # Avoid division by zero if all timestamps are identical
        print_debug(f"[{group_key_val}] All timestamps in the group are identical after processing. Cannot scale meaningfully. Assigning all to cluster 0.")
        df_location['cluster_id_local'] = 0
        total_clusters_count = 1
        inconsistent_clusters_count = 0
        inconsistent_details_list = []
        # Check for inconsistency even in this single cluster
        if len(df_location['common_name'].unique()) > 1:
            inconsistent_clusters_count = 1
            species_in_cluster = df_location['common_name'].unique().tolist()
            inconsistent_details_list.append({
                "cluster_id": 0,
                "species_list": species_in_cluster,
                "count": len(df_location)
            })
        print_debug(f"[{group_key_val}] Assigned to single cluster. Total Clusters={total_clusters_count}, Inconsistent={inconsistent_clusters_count}")
        return df_location, total_clusters_count, inconsistent_clusters_count, records_in_group_after_dropna, inconsistent_details_list, False

    eps_scaled = eps_seconds / scaler.scale_[0] 
    print_debug(f"[{group_key_val}] Scaler std dev (scale_[0]): {scaler.scale_[0]}. Original eps_seconds: {eps_seconds}. Scaled EPS for DBSCAN: {eps_scaled}")

    # Run DBSCAN
    try:
        dbscan = DBSCAN(eps=eps_scaled, min_samples=min_samples_val)
        df_location['cluster_id_local'] = dbscan.fit_predict(scaled_features)
    except Exception as e:
        print_debug(f"[{group_key_val}] Error during DBSCAN execution: {e}. Skipping DBSCAN for this group.")
        # Attempt to return a DataFrame with original columns + an empty cluster_id_local if possible
        df_location['cluster_id_local'] = -1 # Mark all as noise or unclustered
        return df_location, 0, 0, records_in_group_after_dropna, [], False


    # --- Post-processing and Stats ---
    # Count total unique clusters (excluding noise if min_samples > 1, but here min_samples=1 means no noise points from DBSCAN itself)
    # However, we might define noise differently later if needed. For now, all points are in a cluster.
    total_clusters_count = len(df_location['cluster_id_local'].unique())

    # Identify inconsistent clusters (multiple species in one cluster)
    inconsistent_clusters_count = 0
    inconsistent_details_list = []
    for cluster_id, group in df_location.groupby('cluster_id_local'):
        if cluster_id == -1:  # Should not happen with min_samples=1, but good practice
            continue
        unique_species_in_cluster = group['common_name'].nunique()
        if unique_species_in_cluster > 1:
            inconsistent_clusters_count += 1
            species_list = group['common_name'].unique().tolist()
            inconsistent_details_list.append({
                "cluster_id": int(cluster_id), # Ensure it's a Python int for JSON
                "species_list": species_list,
                "count": len(group)
            })
    
    print_debug(f"[{group_key_val}] Original DBSCAN: Total Clusters={total_clusters_count}, Inconsistent={inconsistent_clusters_count}")
    
    # total_animal_records_in_group is the number of records *used for clustering* in this group
    total_animal_records_in_group = records_in_group_after_dropna 
    
    return df_location, total_clusters_count, inconsistent_clusters_count, total_animal_records_in_group, inconsistent_details_list, False # False for used_image_model

def run_dbscan_with_images(location_df_original, image_folder_path, location_id_str, general_eps_value):
    print(f"\n--- Attempting IMAGE-BASED DBSCAN for Group: {location_id_str} ---")
    if not PIL_AVAILABLE or not TENSORFLOW_AVAILABLE:
        print(f"[{location_id_str}] Image processing libraries (PIL/TensorFlow) not available. Falling back to original DBSCAN.")
        return run_dbscan_on_location(location_df_original.copy(), general_eps_value, location_id_str)

    base_model = load_resnet_model()
    if base_model is None:
        print(f"[{location_id_str}] ResNet50 model not loaded. Falling back to original DBSCAN.")
        return run_dbscan_on_location(location_df_original.copy(), general_eps_value, location_id_str)

    location_df = location_df_original.copy() 
    species_column_name = 'common_name'

    image_size = (224, 224)
    image_arrays = []
    image_filenames = [] 

    print(f"[{location_id_str}] Searching for images in folder: {image_folder_path}")
    found_images_count = 0
    for root, _, files in os.walk(image_folder_path): 
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')) and not filename.startswith('._'):
                file_path = os.path.join(root, filename)
                try:
                    image = Image.open(file_path).convert('RGB')
                    image = image.resize(image_size)
                    numpydata = img_to_array(image) 
                    image_arrays.append(numpydata)
                    image_filenames.append(filename) 
                    found_images_count +=1
                except Exception as e:
                    print(f"[{location_id_str}] Skipping {file_path} during image load: {e}")
    
    print(f"[{location_id_str}] Found and attempted to load {found_images_count} images.")

    if not image_arrays:
        print(f"[{location_id_str}] No images loaded successfully from {image_folder_path}. Falling back to original DBSCAN.")
        return run_dbscan_on_location(location_df, general_eps_value, location_id_str)

    all_images_array = np.stack(image_arrays)
    all_images_array_preprocessed = preprocess_input(all_images_array) 

    print(f"[{location_id_str}] Extracting features from {all_images_array_preprocessed.shape[0]} images...")
    image_features_extracted = base_model.predict(all_images_array_preprocessed, batch_size=32, verbose=0) 
    
    df_image_features = pd.DataFrame(image_features_extracted)
    df_image_features['filename'] = image_filenames 

    if 'filename' not in location_df.columns:
        print(f"[{location_id_str}] 'filename' column missing in CSV data for this location. Cannot merge. Falling back.")
        return run_dbscan_on_location(location_df, general_eps_value, location_id_str)

    print(f"[{location_id_str}] Merging {len(location_df)} CSV records with {len(df_image_features)} image features on 'filename'.")
    merged_df = pd.merge(location_df, df_image_features, on='filename', how='inner')
    print(f"[{location_id_str}] Shape of merged_df: {merged_df.shape}")

    if merged_df.empty:
        print(f"[{location_id_str}] Merged DataFrame is empty (no common filenames or other issue). Falling back.")
        return run_dbscan_on_location(location_df, general_eps_value, location_id_str) # Fallback with original location_df

    min_timestamp_in_merged = merged_df['timestamp'].min() # Recalculate for merged subset
    merged_df['timestamp_seconds'] = (merged_df['timestamp'] - min_timestamp_in_merged).dt.total_seconds()
    merged_df[species_column_name] = merged_df[species_column_name].fillna('Unknown_Placeholder')
    merged_df['common_name_id'] = merged_df[species_column_name].astype('category').cat.codes
    
    non_image_features_df = merged_df[['timestamp_seconds', 'common_name_id']].copy()
    
    # Image features are all columns from ResNet output (0 to 2047 typically)
    # These are now columns with integer names in df_image_features, carried to merged_df
    image_feature_column_indices = [col for col in merged_df.columns if isinstance(col, int)]
    if not image_feature_column_indices:
        print(f"[{location_id_str}] No image feature columns (expected integers) found in merged_df. Falling back.")
        return run_dbscan_on_location(location_df, general_eps_value, location_id_str)

    actual_image_features_df = merged_df[image_feature_column_indices].copy()

    scaler_non_image = StandardScaler()
    scaled_non_image = scaler_non_image.fit_transform(non_image_features_df) * 5000 

    scaler_image = StandardScaler()
    scaled_image_features = scaler_image.fit_transform(actual_image_features_df)

    pca_components = min(8, scaled_image_features.shape[1]) 
    if pca_components < 1 : 
        if scaled_image_features.shape[1] == 0:
             print(f"[{location_id_str}] Zero image features after scaling. Falling back.")
             return run_dbscan_on_location(location_df, general_eps_value, location_id_str)
        print(f"[{location_id_str}] Not enough image features ({scaled_image_features.shape[1]}) for PCA. Using unreduced.")
        pca_image_features = scaled_image_features 
    else:
        pca = PCA(n_components=pca_components)
        pca_image_features = pca.fit_transform(scaled_image_features)
        print(f"[{location_id_str}] PCA applied, image features shape: {pca_image_features.shape}")

    combined_features = np.hstack([scaled_non_image, pca_image_features])
    print(f"[{location_id_str}] Combined features shape for DBSCAN: {combined_features.shape}")

    image_model_eps = 25 
    image_model_min_samples = 1

    dbscan_image_model = DBSCAN(eps=image_model_eps, min_samples=image_model_min_samples)
    cluster_labels = dbscan_image_model.fit_predict(combined_features)
    merged_df['cluster_id_local'] = cluster_labels
    
    filename_to_cluster_map = pd.Series(merged_df.cluster_id_local.values, index=merged_df.filename).to_dict()
    location_df['cluster_id_local'] = location_df['filename'].map(filename_to_cluster_map).fillna(-1).astype(int)

    total_animal_records_in_group = len(location_df) 
    unique_cluster_labels_found = np.unique(location_df['cluster_id_local'][location_df['cluster_id_local'] != -1])
    total_clusters_found = len(unique_cluster_labels_found)
    
    inconsistent_clusters_count = 0
    inconsistent_details_list = []

    if total_clusters_found > 0:
        for cluster_val in unique_cluster_labels_found:
            cluster_subset_df = location_df[location_df['cluster_id_local'] == cluster_val]
            if not cluster_subset_df.empty: 
                unique_species_in_cluster = cluster_subset_df[species_column_name].nunique()
                if unique_species_in_cluster > 1:
                    inconsistent_clusters_count += 1
                    species_list = cluster_subset_df[species_column_name].unique().tolist()
                    inconsistent_details_list.append({
                        "cluster_id": int(cluster_val),
                        "species_list": species_list,
                        "count": len(cluster_subset_df)
                    })
    print(f"[{location_id_str}] Image-based DBSCAN results: Total Clusters={total_clusters_found}, Inconsistent={inconsistent_clusters_count}")
    return location_df, total_clusters_found, inconsistent_clusters_count, total_animal_records_in_group, inconsistent_details_list

def main():
    print_debug("--- ORIGINAL main() function CALLED (AddBack Phase 3) ---")
    # Ensure all arguments are captured for debugging
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    eps_arg = sys.argv[3]
    # photo_paths_json_arg might not always be present if not using images
    photo_paths_json_arg = sys.argv[4] if len(sys.argv) > 4 else "{}"

    print_debug(f"Script arguments received: input_file='{input_file}', output_file='{output_file}', eps='{eps_arg}', photo_paths_json (arg4)='{photo_paths_json_arg}'")

    try:
        eps_value_arg = float(eps_arg)
        print_debug(f"Successfully parsed EPS value: {eps_value_arg}")
    except ValueError:
        print_debug(f"PYTHON_SCRIPT_ERROR: Invalid EPS value '{eps_arg}'. Must be a number.")
        # ... (error handling for invalid EPS) ...
        sys.exit(1)
    
    # photo_paths = json.loads(photo_paths_json_arg) if photo_paths_json_arg else {}
    # print_debug(f"Parsed photo_paths_json: {photo_paths}")


    print_debug(f"PYTHON SCRIPT: Using FINAL EPS value: {eps_value_arg} for processing.")

    initial_row_count = 0 # Initialize in case CSV read fails early
    df = None # Initialize df

    try:
        try:
            df = pd.read_csv(input_file)
            initial_row_count = len(df)
            print_debug(f"Successfully read CSV. Initial records: {initial_row_count}")
        except FileNotFoundError:
            print_debug(f"PYTHON_SCRIPT_ERROR: Input file '{input_file}' not found.")
            # Construct and print JSON error output
            error_output = {
                "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
                "clusters": {"message": f"Error: Input file not found: {input_file}", "summary": {}, "details_by_group": {}},
                "error": f"File not found: {input_file}"
            }
            print(json.dumps(error_output)) # Print JSON to stdout
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print_debug(f"PYTHON_SCRIPT_ERROR: Input file '{input_file}' is empty or unparseable.")
            error_output = {
                "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
                "clusters": {"message": f"Error: CSV file is empty or unparseable: {input_file}", "summary": {}, "details_by_group": {}},
                "error": f"CSV file is empty or unparseable: {input_file}"
            }
            print(json.dumps(error_output)) # Print JSON to stdout
            sys.exit(1)
        except Exception as e:
            # Generic CSV read error
            print_debug(f"Error reading CSV: {str(e)}") # Your original debug
            # Let the outer generic Exception catch this to send structured JSON
            raise RuntimeError(f"Error reading CSV: {str(e)}") from e

        print_debug(f"DEBUG: Initial columns from CSV: {df.columns.tolist()}")

        # --- Pre-processing based on your provided logic ---
        if 'deployment_id' not in df.columns:
            print_debug("PYTHON_SCRIPT_ERROR: CSV must contain a 'deployment_id' column for grouping.")
            # Construct and print JSON error output
            error_output = {
                "output_file": os.path.basename(output_file),
                "clusters": {"message": "Error: Missing 'deployment_id' column.", "summary": {}, "details_by_group": {}},
                "error": "Missing 'deployment_id' column."
            }
            print(json.dumps(error_output))
            sys.exit(1)
            
        df.rename(columns={'deployment_id': 'photo_location'}, inplace=True)
        df['photo_location'] = df['photo_location'].astype(str).str.strip().str[-3:]
        print_debug(f"Unique photo_location values after processing: {df['photo_location'].unique()}")
        # --- End of your pre-processing ---

        # --- Global 'timestamp' column check and initial processing ---
        if 'timestamp' not in df.columns:
            print_debug("PYTHON_SCRIPT_ERROR: CSV must contain a 'timestamp' column.")
            error_output = {
                "output_file": os.path.basename(output_file),
                "clusters": {"message": "Error: Missing 'timestamp' column.", "summary": {}, "details_by_group": {}},
                "error": "Missing 'timestamp' column."
            }
            print(json.dumps(error_output))
            sys.exit(1)
        
        # This was the problematic global drop. It's now handled per group.
        # df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # records_before_global_drop = len(df)
        # df.dropna(subset=['timestamp'], inplace=True)
        # print_debug(f"Global timestamp drop: {records_before_global_drop - len(df)} rows removed.")

        if df.empty:
            print_debug("DataFrame is empty after initial read or basic processing (before grouping). No data to cluster.")
            # Fall through to create empty output and JSON summary
        
        print_debug(f"DEBUG: Columns in main 'df' before groupby: {df.columns.tolist()}")

        all_photo_location_stats = {}
        first_records_list = []
        
        grouping_column = 'photo_location'
        
        # Check if df is not empty before attempting to group
        if not df.empty and grouping_column in df.columns and not df[grouping_column].empty:
            for group_key_val, group_data_df in df.groupby(grouping_column):
                processed_df_group, total_c, inconsistent_c, total_a, inconsistent_d, _ = \
                    run_dbscan_on_location(group_data_df.copy(), eps_value_arg, str(group_key_val))
                
                all_photo_location_stats[str(group_key_val)] = {
                    "total_clusters": total_c,
                    "inconsistent_clusters_count": inconsistent_c,
                    "total_animal_records": total_a,
                    "inconsistent_details": inconsistent_d,
                    "used_image_model": False # Explicitly state image model was not used here
                }
                
                if not processed_df_group.empty and 'cluster_id_local' in processed_df_group.columns:
                    df_group_sorted = processed_df_group.sort_values(by=['cluster_id_local', 'timestamp'])
                    first_records_from_group = df_group_sorted[df_group_sorted['cluster_id_local'] != -1].groupby('cluster_id_local').first().reset_index()
                    if not first_records_from_group.empty:
                        first_records_list.append(first_records_from_group)
        elif not df.empty: # Fallback: Process the entire DataFrame as a single group if not empty
            print_debug(f"Warning: Grouping column '{grouping_column}' not found or empty. Processing all data as one group.")
            processed_df_all, total_c, inconsistent_c, total_a, inconsistent_d, _ = \
                run_dbscan_on_location(df.copy(), eps_value_arg, "ALL_DATA") # Pass copy
            
            all_photo_location_stats["ALL_DATA"] = {
                "total_clusters": total_c,
                "inconsistent_clusters_count": inconsistent_c,
                "total_animal_records": total_a,
                "inconsistent_details": inconsistent_d,
                "used_image_model": False
            }
            
            if not processed_df_all.empty and 'cluster_id_local' in processed_df_all.columns:
                df_all_sorted = processed_df_all.sort_values(by=['cluster_id_local', 'timestamp'])
                first_records_from_all = df_all_sorted[df_all_sorted['cluster_id_local'] != -1].groupby('cluster_id_local').first().reset_index()
                if not first_records_from_all.empty:
                    first_records_list.append(first_records_from_all)
        else: # df was empty from the start or after timestamp processing
             print_debug("DataFrame was empty, no groups to process.")


        final_first_records_df = pd.DataFrame() # Initialize as empty
        if first_records_list:
            final_first_records_df = pd.concat(first_records_list, ignore_index=True)
        
        # Determine original columns from the input df *after* potential renames
        # This ensures we try to save columns that actually existed in the input processing flow
        original_input_columns_for_saving = df.columns.tolist() if 'df' in locals() and df is not None and not df.empty else \
                                         (pd.read_csv(input_file).rename(columns={'deployment_id': 'photo_location'}, errors='ignore')).columns.tolist()


        cols_to_save = original_input_columns_for_saving[:] 
        if 'cluster_id_local' not in cols_to_save:
            cols_to_save.append('cluster_id_local')
        
        actual_cols_to_save = [col for col in cols_to_save if col in final_first_records_df.columns]
        
        if 'cluster_id_local' in final_first_records_df.columns and 'cluster_id_local' not in actual_cols_to_save:
             actual_cols_to_save.append('cluster_id_local')
        
        if not final_first_records_df.empty and actual_cols_to_save:
            print_debug(f"DEBUG: Writing final_first_records_df to CSV. Shape: {final_first_records_df.shape}. Columns: {actual_cols_to_save}")
            final_first_records_df.to_csv(output_file, columns=actual_cols_to_save, index=False)
        else: 
            print_debug(f"Warning: final_first_records_df is empty or no columns to save. Creating empty CSV: {output_file}")
            empty_df_cols_for_header = original_input_columns_for_saving[:]
            if 'cluster_id_local' not in empty_df_cols_for_header:
                empty_df_cols_for_header.append('cluster_id_local')
            pd.DataFrame(columns=empty_df_cols_for_header).to_csv(output_file, index=False)

        # Construct the final JSON output for server.js
        final_json_output = {
            "output_file": os.path.basename(output_file),
            "clusters": {
                "message": "Clustering process completed.",
                "summary": {
                    "eps_used": eps_value_arg,
                    "total_records_in_input_csv": initial_row_count,
                    "total_records_processed_for_clustering": sum(group_stats["total_animal_records"] for group_stats in all_photo_location_stats.values()),
                    "total_records_in_output_csv": len(final_first_records_df)
                },
                "details_by_group": all_photo_location_stats
            }
        }
        print_debug("ORIGINAL main(): Attempting to print final_json_output to stdout.")
        print(json.dumps(final_json_output))
        print_debug("ORIGINAL main(): Successfully printed final_json_output.")

    except FileNotFoundError as e:
        print_debug(f"PYTHON_SCRIPT_ERROR in main - File Not Found: {e}")
        error_output = {
            "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
            "clusters": {
                "message": f"Error: Input file not found: {e.filename}",
                "summary": {"error_details": str(e), "error_type": type(e).__name__},
                "details_by_group": {}
            },
            "error": f"File not found: {e.filename}"
        }
        print(json.dumps(error_output))
        sys.exit(1)
    except pd.errors.EmptyDataError as e: # Catch if CSV is empty
        print_debug(f"PYTHON_SCRIPT_ERROR in main - Empty CSV or parse error: {e}")
        error_output = {
            "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
            "clusters": {
                "message": f"Error: CSV file is empty or unparseable: {input_file}",
                "summary": {"error_details": str(e), "error_type": type(e).__name__},
                "details_by_group": {}
            },
            "error": f"CSV file is empty or unparseable: {input_file}"
        }
        print(json.dumps(error_output))
        sys.exit(1)
    except json.JSONDecodeError as e: # Should not happen with current logic, but good to keep
        print_debug(f"PYTHON_SCRIPT_ERROR in main - JSON Decode Error: {e}")
        error_output = {
            "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
            "clusters": {
                "message": f"Error: Invalid JSON encountered: {e.msg}",
                "summary": {"error_details": str(e), "error_type": type(e).__name__},
                "details_by_group": {}
            },
            "error": f"Invalid JSON: {e.msg}"
        }
        print(json.dumps(error_output))
        sys.exit(1)
    except ValueError as e: # Catch specific ValueErrors like missing columns
        print_debug(f"PYTHON_SCRIPT_ERROR in main - ValueError: {e}")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}")
        error_output = {
            "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
            "clusters": {
                "message": f"Data validation error: {str(e)}",
                "summary": {"error_details": str(e), "error_type": type(e).__name__},
                "details_by_group": {}
            },
            "error": f"Python script data error: {str(e)}"
        }
        print(json.dumps(error_output))
        sys.exit(1)
    except Exception as e: # Catch-all for other unexpected errors
        print_debug(f"PYTHON_SCRIPT_ERROR in main - An unexpected error occurred: {type(e).__name__} - {e}")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}")
        error_output = {
            "output_file": os.path.basename(output_file) if 'output_file' in locals() else "error.csv",
            "clusters": {
                "message": f"An unexpected error occurred during script execution: {str(e)}",
                "summary": {"error_details": str(e), "error_type": type(e).__name__},
                "details_by_group": {}
            },
            "error": f"Python script failed: {type(e).__name__} - {str(e)}"
        }
        print(json.dumps(error_output)) 
        sys.exit(1) 

# --- Standard script execution boilerplate (remains unchanged) ---
print_debug("--- Script (AddBack Phase): Reached point BEFORE if __name__ == '__main__' ---")

if __name__ == '__main__':
    print_debug("--- Script (AddBack Phase): ENTERED if __name__ == '__main__' block ---")
    main() # This will now call your ORIGINAL main()
    print_debug("--- Script (AddBack Phase): main() has COMPLETED ---") # This might not be reached if main() exits early or errors
else:
    print_debug("--- Script (AddBack Phase): __name__ IS NOT '__main__' ---")

print_debug("--- Script (AddBack Phase): Reached VERY END of script ---") # Might not be reached 