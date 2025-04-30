# Load necessary library
library(dplyr)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if enough arguments are provided (input, output, minutes)
if (length(args) < 3) {
  stop("Input CSV, Output CSV, and Filter Minutes must be provided.", call. = FALSE)
}

# Get the input, output file paths, and filter minutes
input_csv_path <- args[1]
output_csv_path <- args[2]
filter_minutes_arg <- args[3]

# --- Validate Filter Minutes ---
filter_minutes <- suppressWarnings(as.numeric(filter_minutes_arg))
if (is.na(filter_minutes) || filter_minutes <= 0) {
    stop(paste("Invalid filter minutes provided:", filter_minutes_arg, ". Must be a positive number."), call. = FALSE)
}

# --- Your Filtering Logic ---

# Read the CSV file provided via the command line argument
# Use tryCatch for basic error handling during file reading
FT_Data <- tryCatch({
    read.csv(input_csv_path, stringsAsFactors = FALSE)
}, error = function(e) {
    stop(paste("Error reading CSV file:", e$message), call. = FALSE)
})


# Select relevant columns (adjust if your actual CSV has different columns)
# It's safer to check if columns exist before selecting/deselecting
required_cols <- c("project_id", "filename", "image_id", "wi_taxon_id",
                   "individual_id", "individual_animal_notes", "markings",
                   "cv_confidence", "license", "bounding_boxes", "uncertainty",
                   "is_blank", "animal_recognizable", "location",
                   "deployment_id", "common_name", "timestamp") # Add cols needed for processing

# Find which required columns are actually present
cols_to_remove <- intersect(colnames(FT_Data), c("project_id","filename","image_id","wi_taxon_id",
                                                 "individual_id","individual_animal_notes","markings",
                                                 "cv_confidence","license","bounding_boxes","uncertainty",
                                                 "is_blank","animal_recognizable","location"))

# Check if essential columns for processing exist
if (!"deployment_id" %in% colnames(FT_Data) || !"common_name" %in% colnames(FT_Data) || !"timestamp" %in% colnames(FT_Data)) {
    stop("Essential columns (deployment_id, common_name, timestamp) not found in the CSV.", call. = FALSE)
}

animals <- FT_Data[, !colnames(FT_Data) %in% cols_to_remove, drop = FALSE]


# Create Site column
animals$Site <- animals$deployment_id
animals <- animals %>% mutate(Site = gsub(".*_", "", Site))

# Create Strata column
animals$Strata <- animals$Site
animals <- animals %>% mutate(Strata = recode(Strata, "N01" = "Off-trail", "N02" = "Off-trail", "N03" = "Off-trail", "T01" = "On-trail", "T02" = "On-trail", "T03" = "On-trail", .default = as.character(Strata))) # Added default

# Convert timestamp to datetime object (handle potential errors)
# Use a flexible format or specify the exact one from your CSV
animals$datetime <- tryCatch({
    # Try common formats or the specific one you expect
    # Example: as.POSIXct(animals$timestamp, format="%Y-%m-%d %H:%M:%S", tz="UTC")
    # Example: as.POSIXct(animals$timestamp, format="%m/%d/%Y %H:%M", tz="UTC") # Your original format
    as.POSIXct(animals$timestamp, format="%m/%d/%Y %H:%M", tz="UTC") # Specify timezone
}, warning = function(w) {
    # Handle parsing warnings if needed
}, error = function(e) {
    stop(paste("Error parsing timestamp column:", e$message), call. = FALSE)
})

# Remove rows where datetime parsing failed (optional, depends on desired behavior)
animals <- animals[!is.na(animals$datetime), ]
if (nrow(animals) == 0) {
  stop("No valid timestamp data found after parsing.", call. = FALSE)
}


# Arrange data
animals <- animals %>% arrange(Site, common_name, datetime)

# Calculate time difference (handle potential errors if grouping results in empty groups)
animals <- animals %>%
  group_by(common_name, Site) %>%
  mutate(last_detection_time = ifelse(row_number() == 1, Inf, # Use Inf instead of 999 for clarity
                                      as.numeric(difftime(datetime, lag(datetime), units = "mins")))) %>%
  ungroup()


# Filter based on time difference
animals_filtered <- animals %>%
    filter(last_detection_time > filter_minutes)

# Calculate the final count
result_count <- nrow(animals_filtered)

# --- New: Write the filtered data to the output CSV path ---
tryCatch({
    write.csv(animals_filtered, file = output_csv_path, row.names = FALSE, quote = TRUE)
}, error = function(e) {
    # If writing fails, stop the script so the backend knows something went wrong
    stop(paste("Error writing filtered CSV file:", e$message), call. = FALSE)
})

# --- Output ---
# Print *only* the final count to standard output (for the backend)
cat(result_count) 