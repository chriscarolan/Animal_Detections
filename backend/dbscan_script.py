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
RESNET_PREPROCESS_INPUT = None # For Keras preprocess_input

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print_debug("Pillow (PIL) loaded.")
except ImportError:
    print_debug("Pillow (PIL) not found. Image processing will be disabled.")

try:
    import tensorflow as tf
    tf_version = tuple(map(int, tf.__version__.split('.')))
    if tf_version >= (2, 0, 0):
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input_tf
        # from tensorflow.keras.preprocessing.image import img_to_array # Not directly used in user's new code snippet
        from tensorflow.keras.models import Model # Not directly used in user's new code snippet
    else:
        from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input_tf
        # from keras.preprocessing.image import img_to_array
        # from keras.models import Model
    RESNET_PREPROCESS_INPUT = resnet_preprocess_input_tf
    TENSORFLOW_AVAILABLE = True
    print_debug(f"TensorFlow and Keras components loaded (TF version: {tf.__version__}).")
except ImportError:
    print_debug("TensorFlow or Keras not found. Image-based clustering will be disabled.")
except AttributeError:
    print_debug("Could not determine TensorFlow version. Assuming Keras path might fail.")
    TENSORFLOW_AVAILABLE = False


BASE_RESNET_MODEL = None

# --- Image Processing Parameters (can be adjusted) ---
IMAGE_PROCESSING_FILENAME_COLUMN = 'filename' # Column in CSV linking to image files
IMAGE_TARGET_SIZE = (224, 224)
IMAGE_PCA_COMPONENTS = 8
IMAGE_DBSCAN_EPS = 10.0 # EPS for DBSCAN when using combined image+time features
IMAGE_DBSCAN_MIN_SAMPLES = 1
TIME_COMMON_NAME_FEATURE_WEIGHT = 5000 # Weight for time/common_name features when combined

def load_resnet_model():
    global BASE_RESNET_MODEL
    if not TENSORFLOW_AVAILABLE:
        print_debug("Cannot load ResNet50 model, TensorFlow is not available.")
        return None
    if BASE_RESNET_MODEL is None:
        try:
            BASE_RESNET_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print_debug("ResNet50 model loaded successfully.")
        except Exception as e:
            print_debug(f"Error loading ResNet50 model: {e}")
            BASE_RESNET_MODEL = None # Ensure it's None if loading failed
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

def run_dbscan_on_location(
    df_location_original, 
    eps_seconds_for_time_only, # EPS for time-only DBSCAN
    group_key_val, 
    use_image_processing_flag,
    photo_folder_path_for_group, # Path to the specific photo folder for this group
    min_samples_val=1 # Default min_samples for both modes
):
    print_debug(f"\n--- Running DBSCAN for Group: {group_key_val} ---")
    print_debug(f"    Time-only EPS (if used): {eps_seconds_for_time_only}, Min Samples: {min_samples_val}")
    print_debug(f"    Use Image Processing: {use_image_processing_flag}")
    print_debug(f"    Photo Folder Path: {photo_folder_path_for_group if photo_folder_path_for_group else 'N/A'}")

    df_location = df_location_original.copy()
    # This will be the DataFrame that undergoes clustering and inconsistency checks
    df_for_clustering_and_output = df_location 
    used_image_model_for_this_group = False 
    records_in_group_after_dropna = len(df_location) # Initial count, will be updated after timestamp drop

    # --- Timestamp and Basic Column Prep (common to both paths) ---
    if 'timestamp' not in df_location.columns:
        print_debug(f"[{group_key_val}] 'timestamp' column missing. Cannot proceed with this group.")
        return pd.DataFrame(), 0, 0, 0, [], False

    df_location['timestamp'] = pd.to_datetime(df_location['timestamp'], errors='coerce')
    records_in_group_before_dropna = len(df_location)
    df_location.dropna(subset=['timestamp'], inplace=True)
    records_in_group_after_dropna = len(df_location)
    
    print_debug(f"[{group_key_val}] Timestamp processing: {records_in_group_before_dropna} initial records, "
                f"{records_in_group_after_dropna} records after coercing and dropping NaT timestamps.")

    if records_in_group_after_dropna == 0:
        print_debug(f"[{group_key_val}] No valid timestamp records remaining. Skipping DBSCAN.")
        return pd.DataFrame(), 0, 0, 0, [], False 

    df_location['timestamp_seconds'] = (df_location['timestamp'] - df_location['timestamp'].min()).dt.total_seconds()
    
    if 'common_name' not in df_location.columns:
        print_debug(f"[{group_key_val}] 'common_name' column missing. Adding empty 'common_name' column.")
        df_location['common_name'] = "Unknown"
    df_location['common_name'] = df_location['common_name'].fillna("Unknown")
    df_location['common_name_id'] = df_location['common_name'].astype('category').cat.codes
    # --- End Basic Column Prep ---

    # --- Path Selection: Image-based or Time-based DBSCAN ---
    can_attempt_image_processing = (
        use_image_processing_flag and
        photo_folder_path_for_group and
        os.path.isdir(photo_folder_path_for_group) and
        PIL_AVAILABLE and
        TENSORFLOW_AVAILABLE and
        RESNET_PREPROCESS_INPUT is not None and # Check for the preprocess function
        BASE_RESNET_MODEL is not None and
        IMAGE_PROCESSING_FILENAME_COLUMN in df_location.columns
    )

    if can_attempt_image_processing:
        print_debug(f"[{group_key_val}] Attempting IMAGE-BASED DBSCAN.")
        
        image_arrays = []
        image_filenames_processed = []
        
        # 1. Load images from the specified folder
        print_debug(f"[{group_key_val}] Loading images from: {photo_folder_path_for_group}")
        for filename_img in os.listdir(photo_folder_path_for_group):
            # Check for common image extensions and ignore hidden/system files like '._'
            if filename_img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and not filename_img.startswith('._'):
                file_path_img = os.path.join(photo_folder_path_for_group, filename_img)
                try:
                    image = Image.open(file_path_img).convert('RGB')
                    image = image.resize(IMAGE_TARGET_SIZE)
                    numpydata = np.asarray(image) # asarray is fine
                    image_arrays.append(numpydata)
                    image_filenames_processed.append(filename_img)
                except Exception as e:
                    print_debug(f"[{group_key_val}] Skipping image {filename_img} due to error: {e}")
        
        if not image_arrays:
            print_debug(f"[{group_key_val}] No images successfully loaded from {photo_folder_path_for_group}. Falling back to time-based DBSCAN.")
            # Fall through to time-based DBSCAN logic below (used_image_model_for_this_group remains False)
        else:
            all_images_array = np.stack(image_arrays)
            # Use the globally available RESNET_PREPROCESS_INPUT
            all_images_array_preprocessed = RESNET_PREPROCESS_INPUT(all_images_array) 
            
            # 2. Extract features using ResNet
            print_debug(f"[{group_key_val}] Extracting ResNet features for {len(all_images_array_preprocessed)} images.")
            image_resnet_features = BASE_RESNET_MODEL.predict(all_images_array_preprocessed, batch_size=32, verbose=0)
            
            df_image_features = pd.DataFrame(image_resnet_features) # Columns will be 0, 1, ..., 2047
            df_image_features[IMAGE_PROCESSING_FILENAME_COLUMN] = image_filenames_processed
            
            # 3. Merge image features with location data (df_location)
            # Ensure the filename column in df_location is string type for robust merging
            df_location[IMAGE_PROCESSING_FILENAME_COLUMN] = df_location[IMAGE_PROCESSING_FILENAME_COLUMN].astype(str).str.strip()
            df_image_features[IMAGE_PROCESSING_FILENAME_COLUMN] = df_image_features[IMAGE_PROCESSING_FILENAME_COLUMN].astype(str).str.strip()

            # Use df_location which already has timestamp_seconds and common_name_id
            df_location_merged_with_images = pd.merge(df_location, df_image_features, on=IMAGE_PROCESSING_FILENAME_COLUMN, how='inner')
            
            if df_location_merged_with_images.empty:
                print_debug(f"[{group_key_val}] No matching records after merging CSV data with image features. Falling back to time-based DBSCAN on original df_location.")
                # Fall through to time-based DBSCAN logic below
            else:
                print_debug(f"[{group_key_val}] Successfully merged {len(df_location_merged_with_images)} records with image features.")
                
                # 4. Prepare features for combined DBSCAN
                non_image_features_df = df_location_merged_with_images[['timestamp_seconds', 'common_name_id']].copy()
                scaler_non_image = StandardScaler()
                scaled_non_image_features = scaler_non_image.fit_transform(non_image_features_df) * TIME_COMMON_NAME_FEATURE_WEIGHT
                
                # Image features are columns 0 to N-1 from ResNet output
                num_resnet_features = image_resnet_features.shape[1]
                # Get the actual column names for image features after merge (they are 0, 1, ..., N-1)
                image_feature_column_indices = list(range(num_resnet_features)) 
                actual_image_features_from_merged = df_location_merged_with_images[image_feature_column_indices].copy()

                scaler_image = StandardScaler()
                scaled_image_features = scaler_image.fit_transform(actual_image_features_from_merged)
                
                pca = PCA(n_components=IMAGE_PCA_COMPONENTS)
                pca_transformed_image_features = pca.fit_transform(scaled_image_features)
                
                combined_features_for_dbscan = np.hstack([scaled_non_image_features, pca_transformed_image_features])
                
                # 5. Run DBSCAN on combined features
                print_debug(f"[{group_key_val}] Running DBSCAN on combined features with EPS={IMAGE_DBSCAN_EPS}, MinSamples={IMAGE_DBSCAN_MIN_SAMPLES}")
                dbscan_model = DBSCAN(eps=IMAGE_DBSCAN_EPS, min_samples=IMAGE_DBSCAN_MIN_SAMPLES)
                cluster_labels = dbscan_model.fit_predict(combined_features_for_dbscan)
                
                df_location_merged_with_images['cluster_id_local'] = cluster_labels
                df_for_clustering_and_output = df_location_merged_with_images # Update the main df for this group
                used_image_model_for_this_group = True
                print_debug(f"[{group_key_val}] Image-based DBSCAN complete. Clusters found: {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    
    # --- Fallback or Standard Time-based DBSCAN ---
    if not used_image_model_for_this_group:
        print_debug(f"[{group_key_val}] Using TIME-BASED DBSCAN.")
        # df_for_clustering_and_output is already df_location by default
        if df_for_clustering_and_output.empty or 'timestamp_seconds' not in df_for_clustering_and_output.columns or df_for_clustering_and_output[['timestamp_seconds']].isnull().all().all():
             print_debug(f"[{group_key_val}] No valid data for time-based DBSCAN. Skipping.")
             # records_in_group_after_dropna would be from the initial timestamp check
             return pd.DataFrame(), 0, 0, records_in_group_after_dropna, [], False

        features_time_only = df_for_clustering_and_output[['timestamp_seconds']]
        scaler_time_only = StandardScaler()
        try:
            scaled_features_time_only = scaler_time_only.fit_transform(features_time_only)
        except ValueError: 
            print_debug(f"[{group_key_val}] ValueError during scaling for time-only DBSCAN (e.g., only one sample). Assigning all to cluster 0.")
            df_for_clustering_and_output['cluster_id_local'] = 0
        else:
            if scaler_time_only.scale_[0] == 0: 
                print_debug(f"[{group_key_val}] All timestamps identical for time-only DBSCAN. Assigning all to cluster 0.")
                df_for_clustering_and_output['cluster_id_local'] = 0
            else:
                eps_for_dbscan_time_only = eps_seconds_for_time_only / scaler_time_only.scale_[0]
                print_debug(f"[{group_key_val}] Running DBSCAN on time features with Scaled EPS={eps_for_dbscan_time_only}, MinSamples={min_samples_val}")
                dbscan_model_time_only = DBSCAN(eps=eps_for_dbscan_time_only, min_samples=min_samples_val)
                cluster_labels_time_only = dbscan_model_time_only.fit_predict(scaled_features_time_only)
                df_for_clustering_and_output['cluster_id_local'] = cluster_labels_time_only
        
        if 'cluster_id_local' in df_for_clustering_and_output.columns:
            print_debug(f"[{group_key_val}] Time-based DBSCAN complete. Clusters found: {len(np.unique(df_for_clustering_and_output['cluster_id_local'])) - (1 if -1 in df_for_clustering_and_output['cluster_id_local'].unique() else 0)}")
        else: # Should not happen if logic above is correct
            print_debug(f"[{group_key_val}] 'cluster_id_local' not assigned after time-based DBSCAN attempt.")
            df_for_clustering_and_output['cluster_id_local'] = -1 # Default to noise if something went wrong

    # --- Post-DBSCAN Processing (Inconsistency Check, etc.) ---
    if df_for_clustering_and_output.empty or 'cluster_id_local' not in df_for_clustering_and_output.columns:
        print_debug(f"[{group_key_val}] No data or cluster IDs after DBSCAN. Returning empty results for this group.")
        return pd.DataFrame(), 0, 0, records_in_group_after_dropna, [], used_image_model_for_this_group

    total_clusters_count = len(df_for_clustering_and_output[df_for_clustering_and_output['cluster_id_local'] != -1]['cluster_id_local'].unique())
    inconsistent_clusters_count = 0
    inconsistent_details_list = []

    for cluster_id, cluster_group_df in df_for_clustering_and_output[df_for_clustering_and_output['cluster_id_local'] != -1].groupby('cluster_id_local'):
        unique_species_in_cluster = cluster_group_df['common_name'].unique()
        if len(unique_species_in_cluster) > 1:
            inconsistent_clusters_count += 1
            inconsistent_details_list.append({
                "cluster_id": int(cluster_id),
                "species_list": unique_species_in_cluster.tolist(),
                "count": len(cluster_group_df)
            })
    
    print_debug(f"[{group_key_val}] Final counts: Total Clusters={total_clusters_count}, Inconsistent={inconsistent_clusters_count}, Animals Processed in group={records_in_group_after_dropna}")
    # Return the DataFrame that has the 'cluster_id_local' column
    return df_for_clustering_and_output, total_clusters_count, inconsistent_clusters_count, records_in_group_after_dropna, inconsistent_details_list, used_image_model_for_this_group

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
                    numpydata = np.asarray(image) 
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
    all_images_array_preprocessed = RESNET_PREPROCESS_INPUT(all_images_array) 

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
    print_debug("--- DBSCAN SCRIPT (with conditional image processing) CALLED ---")
    
    # Expecting 5 arguments now: input_file, output_file, eps_arg_for_time_only, location_processing_config_json, (optional old photo_paths_json - will ignore for now)
    if len(sys.argv) < 5: 
        print_debug("PYTHON_SCRIPT_ERROR: Insufficient arguments. Expected: input_csv, output_csv, eps_for_time_only, location_processing_config_json")
        error_output = {"error": "Insufficient script arguments.", "details": "Expected input_csv, output_csv, eps_for_time_only, location_processing_config_json"}
        print(json.dumps({"output_file": "error.csv", "clusters": {"message": error_output["error"], "summary": {}, "details_by_group": {}}, "error_details_full": error_output}))
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    eps_arg_for_time_only_cli = sys.argv[3]
    location_processing_config_json_arg = sys.argv[4] 

    try:
        eps_value_for_time_only = float(eps_arg_for_time_only_cli)
    except ValueError:
        print_debug(f"PYTHON_SCRIPT_ERROR: Invalid EPS value for time-only DBSCAN '{eps_arg_for_time_only_cli}'.")
        # ... (construct and print error JSON) ...
        sys.exit(1)

    location_processing_config = {}
    try:
        location_processing_config = json.loads(location_processing_config_json_arg)
        print_debug(f"Parsed location_processing_config: {location_processing_config}")
    except json.JSONDecodeError as e:
        print_debug(f"PYTHON_SCRIPT_WARNING: Could not parse location_processing_config_json: {e}. No image processing will be enabled. JSON received: {location_processing_config_json_arg}")
        # location_processing_config remains empty, so all groups will default to no image processing

    # Load ResNet model once if TF is available and PIL is available
    if TENSORFLOW_AVAILABLE and PIL_AVAILABLE:
        load_resnet_model() # Attempts to load/set BASE_RESNET_MODEL

    initial_row_count = 0
    df = pd.DataFrame() # Initialize df

    try:
        df = pd.read_csv(input_file)
        initial_row_count = len(df)
        print_debug(f"Successfully read CSV. Initial records: {initial_row_count}")
    except FileNotFoundError:
        print_debug(f"PYTHON_SCRIPT_ERROR: Input file '{input_file}' not found.")
        # ... (construct and print error JSON) ...
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print_debug(f"PYTHON_SCRIPT_ERROR: Input file '{input_file}' is empty.")
        # ... (construct and print error JSON) ...
        sys.exit(1)
    except Exception as e:
        print_debug(f"PYTHON_SCRIPT_ERROR: Error reading CSV '{input_file}': {e}")
        # ... (construct and print error JSON) ...
        sys.exit(1)
        
    print_debug(f"DEBUG: Initial columns from CSV: {df.columns.tolist()}")

    # --- Pre-processing for 'photo_location' ---
    if 'deployment_id' not in df.columns:
        print_debug("PYTHON_SCRIPT_ERROR: CSV must contain a 'deployment_id' column for grouping.")
        # ... (construct and print error JSON) ...
        sys.exit(1)
            
    df.rename(columns={'deployment_id': 'photo_location'}, inplace=True)
    df['photo_location'] = df['photo_location'].astype(str).str.strip().str[-3:]
    print_debug(f"Unique photo_location values after processing: {df['photo_location'].unique().tolist()}")
    
    if IMAGE_PROCESSING_FILENAME_COLUMN not in df.columns:
        print_debug(f"WARNING: The CSV does not contain the expected filename column '{IMAGE_PROCESSING_FILENAME_COLUMN}'. Image-based processing will not be possible for any group.")
        # This doesn't stop the script, groups will just default to time-based.

    # --- Main processing loop ---
    all_photo_location_stats = {}
    processed_group_dataframes_list = [] # Store processed DFs from each group
    grouping_column = 'photo_location'
    total_records_processed_for_clustering_sum = 0

    if grouping_column in df.columns and not df[grouping_column].empty:
        for group_key_val_str, group_data_df in df.groupby(grouping_column):
            group_key_val = str(group_key_val_str) 
            
            group_config = location_processing_config.get(group_key_val, {})
            use_image_for_group_str = group_config.get('use_image', 'no') # Get 'yes' or 'no'
            use_image_for_group_bool = use_image_for_group_str.lower() == 'yes'
            photo_folder_for_group = group_config.get('path', None)

            # Further checks if image processing is intended
            if use_image_for_group_bool:
                if not (photo_folder_for_group and os.path.isdir(photo_folder_for_group)):
                    print_debug(f"WARNING for group {group_key_val}: 'use_image' is 'yes' but photo folder path ('{photo_folder_for_group}') is invalid or missing. Defaulting to time-based DBSCAN.")
                    use_image_for_group_bool = False
                elif not (PIL_AVAILABLE and TENSORFLOW_AVAILABLE and BASE_RESNET_MODEL):
                    print_debug(f"WARNING for group {group_key_val}: 'use_image' is 'yes' but required libraries (PIL, TF) or ResNet model are not available. Defaulting to time-based DBSCAN.")
                    use_image_for_group_bool = False
                elif IMAGE_PROCESSING_FILENAME_COLUMN not in group_data_df.columns:
                    print_debug(f"WARNING for group {group_key_val}: 'use_image' is 'yes' but the CSV data for this group is missing the filename column '{IMAGE_PROCESSING_FILENAME_COLUMN}'. Defaulting to time-based DBSCAN.")
                    use_image_for_group_bool = False
            
            processed_df_group, total_c, inconsistent_c, total_a, inconsistent_d, image_model_was_used_flag = \
                run_dbscan_on_location(
                    group_data_df.copy(), # Pass a copy to avoid SettingWithCopyWarning
                    eps_value_for_time_only,
                    group_key_val,
                    use_image_for_group_bool, # Pass the boolean flag
                    photo_folder_for_group
                )
            
            total_records_processed_for_clustering_sum += total_a
            all_photo_location_stats[group_key_val] = {
                "total_clusters": total_c,
                "inconsistent_clusters_count": inconsistent_c,
                "total_animal_records": total_a, # This is count after per-group timestamp drop
                "inconsistent_details": inconsistent_d,
                "used_image_model": image_model_was_used_flag
            }

            if not processed_df_group.empty:
                 # Ensure the processed_df_group has the necessary columns for concat, even if it's just original cols + cluster_id_local
                processed_group_dataframes_list.append(processed_df_group)
    else:
        print_debug(f"Warning: Grouping column '{grouping_column}' not found or empty. No group-wise processing done.")
        # Potentially handle as a single group if desired, or output empty results.
        # For now, it will result in an empty final_first_records_df if no groups are processed.

    final_clustered_df = pd.DataFrame()
    if processed_group_dataframes_list:
        try:
            # Concatenate all processed group DataFrames. They should all have 'cluster_id_local'.
            # Select common columns or handle schema differences if they arise.
            # For now, assume they can be concatenated directly if they come from the same source + cluster_id_local
            final_clustered_df = pd.concat(processed_group_dataframes_list, ignore_index=True)
        except Exception as e:
            print_debug(f"Error concatenating processed group dataframes: {e}")
            # Fallback: create an empty df or handle error more gracefully

    # Extract first records from the combined and clustered DataFrame
    final_first_records_df = pd.DataFrame()
    if not final_clustered_df.empty and 'cluster_id_local' in final_clustered_df.columns and 'timestamp' in final_clustered_df.columns:
        # Ensure timestamp is datetime for sorting, if not already
        final_clustered_df['timestamp'] = pd.to_datetime(final_clustered_df['timestamp'], errors='coerce')
        df_sorted_for_first_records = final_clustered_df.sort_values(by=['photo_location', 'cluster_id_local', 'timestamp'])
        
        # Get the first record of each actual cluster (exclude noise points)
        # Group by photo_location and then by cluster_id_local to get first per cluster *per location*
        final_first_records_df = df_sorted_for_first_records[
            df_sorted_for_first_records['cluster_id_local'] != -1
        ].groupby(['photo_location', 'cluster_id_local']).first().reset_index()
    
    # --- Save to output CSV ---
    # Determine original columns from the input df *after* potential renames
    original_input_columns_for_saving = df.columns.tolist() if df is not None and not df.empty else \
                                     (pd.read_csv(input_file).rename(columns={'deployment_id': 'photo_location'}, errors='ignore')).columns.tolist()

    cols_to_save = original_input_columns_for_saving[:] 
    if 'cluster_id_local' not in cols_to_save and 'cluster_id_local' in final_first_records_df.columns:
        cols_to_save.append('cluster_id_local')
    
    # Ensure we only try to save columns that actually exist in final_first_records_df
    actual_cols_to_save = [col for col in cols_to_save if col in final_first_records_df.columns]
    
    if not final_first_records_df.empty and actual_cols_to_save:
        print_debug(f"DEBUG: Writing final_first_records_df to CSV. Shape: {final_first_records_df.shape}. Columns: {actual_cols_to_save}")
        final_first_records_df.to_csv(output_file, columns=actual_cols_to_save, index=False)
    else: 
        print_debug(f"Warning: final_first_records_df is empty or no columns to save. Creating empty CSV: {output_file}")
        empty_df_cols_for_header = original_input_columns_for_saving[:]
        if 'cluster_id_local' not in empty_df_cols_for_header: # Add if it was expected
            empty_df_cols_for_header.append('cluster_id_local')
        pd.DataFrame(columns=empty_df_cols_for_header).to_csv(output_file, index=False)

    # --- Construct final JSON output ---
    final_json_output = {
        "output_file": os.path.basename(output_file),
        "clusters": {
            "message": "Clustering process completed.",
            "summary": {
                "eps_used_for_time_only_dbscan": eps_value_for_time_only,
                "eps_used_for_image_dbscan": IMAGE_DBSCAN_EPS, # Add this
                "pca_components_for_image_dbscan": IMAGE_PCA_COMPONENTS, # Add this
                "time_feature_weight_for_image_dbscan": TIME_COMMON_NAME_FEATURE_WEIGHT, # Add this
                "total_records_in_input_csv": initial_row_count,
                "total_records_processed_for_clustering": total_records_processed_for_clustering_sum,
                "total_records_in_output_csv": len(final_first_records_df)
            },
            "details_by_group": all_photo_location_stats # This now contains 'used_image_model'
        }
    }
    print_debug("Attempting to print final_json_output to stdout.")
    print(json.dumps(final_json_output))
    print_debug("Successfully printed final_json_output.")

# --- Standard script execution boilerplate (remains unchanged) ---
print_debug("--- Script (AddBack Phase): Reached point BEFORE if __name__ == '__main__' ---")

if __name__ == '__main__':
    print_debug("--- Script (AddBack Phase): ENTERED if __name__ == '__main__' block ---")
    main() # This will now call your ORIGINAL main()
    print_debug("--- Script (AddBack Phase): main() has COMPLETED ---") # This might not be reached if main() exits early or errors
else:
    print_debug("--- Script (AddBack Phase): __name__ IS NOT '__main__' ---")

print_debug("--- Script (AddBack Phase): Reached VERY END of script ---") # Might not be reached 