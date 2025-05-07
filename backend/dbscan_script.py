import pandas as pd
from sklearn.cluster import DBSCAN
# StandardScaler might not be needed if we use unscaled features as per your snippet
# from sklearn.preprocessing import StandardScaler
import sys
import json # To output results in JSON format
import re   # For replicating the Site column creation
import os   # For file operations
import numpy as np
from collections import Counter

def create_site_column(df):
    """Replicates the Site column creation from the R script."""
    # Assuming 'deployment_id' might be the original name before renaming to 'photo_location'
    # Or, if 'photo_location' is always present and is the basis for 'Site', adjust accordingly.
    # For now, let's assume it might operate on the original 'deployment_id' if it exists,
    # or you might call it with the new 'photo_location' column if that's intended.
    # This function is not directly part of the DBSCAN logic below but might be used elsewhere.
    id_col_for_site = 'deployment_id' # or 'photo_location' depending on when/how it's used
    if id_col_for_site not in df.columns:
        # Try the other common name if one was renamed
        if 'photo_location' in df.columns and id_col_for_site == 'deployment_id':
            id_col_for_site = 'photo_location'
        elif 'deployment_id' in df.columns and id_col_for_site == 'photo_location':
             id_col_for_site = 'deployment_id'
        else:
            print(f"Warning: Column '{id_col_for_site}' not found for Site creation. Skipping Site column.", file=sys.stderr)
            return df # Return df unmodified

    df['Site'] = df[id_col_for_site].astype(str).apply(lambda x: re.sub(r'.*_', '', x))
    return df

# Modified to include group_key_for_debug for targeted print statements
def run_dbscan_on_location(df_group, group_key_for_debug="UNKNOWN_GROUP", min_samples_val=1):
    print(f"\n--- Debugging Group: {group_key_for_debug} ---", file=sys.stderr)
    print(f"Initial group shape: {df_group.shape}", file=sys.stderr)
    print(f"DEBUG: Columns in df_group at entry of run_dbscan_on_location: {df_group.columns.tolist()}", file=sys.stderr)

    # Use 'common_name' instead of 'animal_common_name'
    species_column_name = 'common_name' 

    if df_group.empty or species_column_name not in df_group.columns or 'timestamp' not in df_group.columns:
        print(f"[{group_key_for_debug}] Group is empty or missing essential columns at start.", file=sys.stderr)
        if species_column_name not in df_group.columns:
            print(f"[{group_key_for_debug}] DEBUG: Column '{species_column_name}' IS MISSING.", file=sys.stderr)
        if 'timestamp' not in df_group.columns:
            print(f"[{group_key_for_debug}] DEBUG: Column 'timestamp' IS MISSING.", file=sys.stderr)
        # Return the original (or a copy) df_group and zero stats
        return df_group.copy() if df_group is not None else pd.DataFrame(), 0, 0, 0, []

    current_group_df = df_group.copy()
    current_group_df.dropna(subset=['timestamp'], inplace=True)
    print(f"[{group_key_for_debug}] Shape after dropping NaT timestamps: {current_group_df.shape}", file=sys.stderr)

    if current_group_df['timestamp'].empty:
        print(f"[{group_key_for_debug}] No valid timestamps after internal dropna.", file=sys.stderr)
        # Return current_group_df (which is empty of valid timestamps) and zero stats for clusters
        return current_group_df, 0, 0, len(current_group_df), [] 

    min_timestamp_in_group = current_group_df['timestamp'].min()
    current_group_df['timestamp_seconds'] = (current_group_df['timestamp'] - min_timestamp_in_group).dt.total_seconds()
    print(f"[{group_key_for_debug}] Min timestamp: {min_timestamp_in_group}", file=sys.stderr)
    print(f"[{group_key_for_debug}] timestamp_seconds (head 5):\n{current_group_df[['timestamp', 'timestamp_seconds']].head()}", file=sys.stderr)

    # Use 'common_name'
    current_group_df[species_column_name] = current_group_df[species_column_name].fillna('Unknown_Placeholder')
    current_group_df['common_name_id'] = current_group_df[species_column_name].astype('category').cat.codes
    print(f"[{group_key_for_debug}] common_name_id (head 5):\n{current_group_df[[species_column_name, 'common_name_id']].head()}", file=sys.stderr)
    print(f"[{group_key_for_debug}] Unique common_name_id values: {current_group_df['common_name_id'].unique()}", file=sys.stderr)


    if current_group_df.shape[0] == 0:
        print(f"[{group_key_for_debug}] Group is empty before feature preparation.", file=sys.stderr)
        return current_group_df, 0,0,0,[] # Return empty current_group_df
        
    # Using UN SCALED features as per your snippet for individual_animal_count
    features_unscaled = current_group_df[['timestamp_seconds', 'common_name_id']].values
    print(f"[{group_key_for_debug}] UN SCALED features (first 5 rows):\n{features_unscaled[:5]}", file=sys.stderr)

    clusters = np.array([-1] * current_group_df.shape[0]) # Default to all noise

    if features_unscaled.shape[0] > 0: # If there are any features to process
        try:
            # Using eps=10 directly on unscaled features
            dbscan = DBSCAN(eps=10, min_samples=min_samples_val) 
            clusters = dbscan.fit_predict(features_unscaled)
            # current_group_df['cluster_id_local'] = clusters # This will be assigned below
        except Exception as e: # Catch any error during DBSCAN
            print(f"[{group_key_for_debug}] DBSCAN error on unscaled features: {e}. Assigning all as noise.", file=sys.stderr)
            # clusters already defaulted to noise
    else:
        print(f"[{group_key_for_debug}] No features to process for DBSCAN.", file=sys.stderr)


    print(f"[{group_key_for_debug}] DBSCAN cluster labels (first 20): {clusters[:20]}", file=sys.stderr)
    current_group_df['cluster_id_local'] = clusters # Assign cluster labels to the DataFrame

    total_animal_records_in_group = len(current_group_df)
    
    # Total clusters (equivalent to your individual_animal_count)
    unique_cluster_labels = np.unique(clusters[clusters != -1]) # Exclude noise points (-1)
    total_clusters_found = len(unique_cluster_labels)
    print(f"[{group_key_for_debug}] Total clusters found (excluding noise): {total_clusters_found}", file=sys.stderr)
    print(f"[{group_key_for_debug}] Unique cluster labels (excluding noise): {unique_cluster_labels}", file=sys.stderr)

    inconsistent_clusters_count = 0
    inconsistent_details = []
    valid_clusters_df = current_group_df[current_group_df['cluster_id_local'] != -1]

    if not valid_clusters_df.empty:
        for cluster_id_val_iter in valid_clusters_df['cluster_id_local'].unique(): # Renamed to avoid conflict
            current_cluster_records = valid_clusters_df[valid_clusters_df['cluster_id_local'] == cluster_id_val_iter]
            # Use 'common_name'
            if current_cluster_records[species_column_name].nunique() > 1:
                inconsistent_clusters_count += 1
                details = []
                for _, row_data in current_cluster_records.iterrows(): # Renamed 'row' to 'row_data'
                    details.append({
                        "common_name": row_data[species_column_name], # Use 'common_name'
                        "timestamp": row_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row_data['timestamp']) else None
                    })
                inconsistent_details.append({
                    "cluster_id_local": int(cluster_id_val_iter),
                    "records": details
                })
    
    print(f"[{group_key_for_debug}] Inconsistent clusters: {inconsistent_clusters_count}", file=sys.stderr)
    print(f"--- End Debugging Group: {group_key_for_debug} ---\n", file=sys.stderr)
    # Return the processed DataFrame along with stats
    return current_group_df, total_clusters_found, inconsistent_clusters_count, total_animal_records_in_group, inconsistent_details

# --- Main Script Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python dbscan_script.py <input_csv_path> <output_csv_path>"}), file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(json.dumps({"error": f"Input file not found: {input_file}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Error reading CSV: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

    print(f"DEBUG: Initial columns from CSV: {df.columns.tolist()}", file=sys.stderr) # ADDED

    # --- Pre-processing based on your new requirements ---
    if 'deployment_id' not in df.columns:
        print(json.dumps({"error": "CSV must contain a 'deployment_id' column for grouping."}), file=sys.stderr)
        sys.exit(1)
        
    df.rename(columns={'deployment_id': 'photo_location'}, inplace=True)
    df['photo_location'] = df['photo_location'].astype(str).str[-3:]
    print(f"Unique photo_location values after processing: {df['photo_location'].unique()}", file=sys.stderr)
    # --- End of new pre-processing ---

    if 'timestamp' not in df.columns:
        print(json.dumps({"error": "CSV must contain a 'timestamp' column."}), file=sys.stderr)
        sys.exit(1)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    initial_row_count = len(df)
    df.dropna(subset=['timestamp'], inplace=True)
    print(f"Global timestamp drop: {initial_row_count - len(df)} rows removed.", file=sys.stderr)

    print(f"DEBUG: Columns in main 'df' before groupby: {df.columns.tolist()}", file=sys.stderr)

    all_photo_location_stats = {} # Changed name
    first_records_list = [] # This will store the first record of each cluster from each group

    grouping_column = 'photo_location'

    if grouping_column in df.columns and not df[grouping_column].empty:
        for group_key_val, group_data_df in df.groupby(grouping_column):
            # Call run_dbscan_on_location and get the processed DataFrame back
            processed_df_group, total_c, inconsistent_c, total_a, inconsistent_d = \
                run_dbscan_on_location(group_data_df, str(group_key_val))
            
            all_photo_location_stats[str(group_key_val)] = {
                "total_clusters": total_c,
                "inconsistent_clusters_count": inconsistent_c,
                "total_animal_records": total_a,
                "inconsistent_details": inconsistent_d
            }
            
            # Use the returned processed_df_group to extract first records
            if not processed_df_group.empty and 'cluster_id_local' in processed_df_group.columns:
                df_group_sorted = processed_df_group.sort_values(by=['cluster_id_local', 'timestamp'])
                first_records_from_group = df_group_sorted[df_group_sorted['cluster_id_local'] != -1].groupby('cluster_id_local').first().reset_index()
                if not first_records_from_group.empty:
                    first_records_list.append(first_records_from_group)
    else:
        # Fallback: Process the entire DataFrame as a single group
        processed_df_all, total_c, inconsistent_c, total_a, inconsistent_d = \
            run_dbscan_on_location(df, "ALL_DATA") # Pass df directly
        
        all_photo_location_stats["ALL_DATA"] = {
            "total_clusters": total_c,
            "inconsistent_clusters_count": inconsistent_c,
            "total_animal_records": total_a,
            "inconsistent_details": inconsistent_d
        }
        
        # Use the returned processed_df_all
        if not processed_df_all.empty and 'cluster_id_local' in processed_df_all.columns:
            df_all_sorted = processed_df_all.sort_values(by=['cluster_id_local', 'timestamp'])
            first_records_from_all = df_all_sorted[df_all_sorted['cluster_id_local'] != -1].groupby('cluster_id_local').first().reset_index()
            if not first_records_from_all.empty:
                first_records_list.append(first_records_from_all)

    # Combine all first records and save to CSV
    if first_records_list:
        final_first_records_df = pd.concat(first_records_list, ignore_index=True)
        # Ensure original columns are prioritized, plus 'cluster_id_local'
        original_input_columns = df.columns.tolist() # These are after renaming deployment_id to photo_location
        
        # Columns to save: original ones + cluster_id_local, if not already there
        cols_to_save = original_input_columns[:] 
        if 'cluster_id_local' not in cols_to_save:
            cols_to_save.append('cluster_id_local')
        
        # Filter out any columns in cols_to_save that are not in final_first_records_df
        actual_cols_to_save = [col for col in cols_to_save if col in final_first_records_df.columns]
        
        # If 'cluster_id_local' is essential and somehow got missed but exists in df, add it back
        if 'cluster_id_local' in final_first_records_df.columns and 'cluster_id_local' not in actual_cols_to_save:
            actual_cols_to_save.append('cluster_id_local')

        if not final_first_records_df.empty and actual_cols_to_save:
            print(f"DEBUG: Writing final_first_records_df to CSV. Shape: {final_first_records_df.shape}. Columns: {actual_cols_to_save}", file=sys.stderr)
            final_first_records_df.to_csv(output_file, columns=actual_cols_to_save, index=False)
        else: # If final_first_records_df is empty or no valid columns to save
            # Create an empty CSV with expected headers (original + cluster_id_local)
            empty_df_cols = original_input_columns[:]
            if 'cluster_id_local' not in empty_df_cols:
                empty_df_cols.append('cluster_id_local')
            pd.DataFrame(columns=empty_df_cols).to_csv(output_file, index=False)
    else:
        # Create an empty CSV with headers if no clusters were formed at all
        empty_df_cols = df.columns.tolist() # Original columns after rename
        if 'cluster_id_local' not in empty_df_cols: 
            empty_df_cols.append('cluster_id_local')
        pd.DataFrame(columns=empty_df_cols).to_csv(output_file, index=False)

    print(json.dumps(all_photo_location_stats)) 