const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process'); // Use execFile for better security/control
const path = require('path');
const fs = require('fs'); // File system module
const crypto = require('crypto'); // For generating unique IDs
const { spawn } = require('child_process'); // Added for spawn method

const app = express();
const port = 3000; // Port the server will listen on

// --- In-memory store for cluster results ---
// Simple store: object mapping resultId to data.
// In a production app, you'd use a database or a more robust caching solution.
const clusterResultsStore = {};
const RESULT_EXPIRY_TIME = 15 * 60 * 1000; // 15 minutes in milliseconds

// --- File Upload Setup (using Multer) ---
// Configure where to store uploaded files temporarily
const baseUploadDir = path.join(__dirname, 'uploads'); // Base for all uploads
if (!fs.existsSync(baseUploadDir)){
    fs.mkdirSync(baseUploadDir, { recursive: true });
}
const filteredOutputDir = path.join(__dirname, 'filtered_outputs');
if (!fs.existsSync(filteredOutputDir)){
    fs.mkdirSync(filteredOutputDir, { recursive: true });
}
const clusteredOutputDir = path.join(__dirname, 'clustered_outputs');
if (!fs.existsSync(clusteredOutputDir)){
    fs.mkdirSync(clusteredOutputDir, { recursive: true });
}

// Configure Multer for dynamic fields
// const multerFields = [ // We'll define this inline for the test
//     { name: 'csvFile', maxCount: 1 },
//     { name: 'photoN01_files', maxCount: 200 },
//     { name: 'photoN02_files', maxCount: 200 },
//     { name: 'photoN03_files', maxCount: 200 },
//     { name: 'photoT01_files', maxCount: 200 },
//     { name: 'photoT02_files', maxCount: 200 },
//     { name: 'photoT03_files', maxCount: 200 }
// ];

// console.log("SERVER.JS: Initializing Multer with these fields:", JSON.stringify(multerFields, null, 2)); // No longer using the variable here

// Temporary storage for multer before organizing
const tempMulterUploads = path.join(baseUploadDir, 'temp_multer');
if (!fs.existsSync(tempMulterUploads)) {
    fs.mkdirSync(tempMulterUploads, { recursive: true });
}
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, tempMulterUploads);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        // Use file.originalname to preserve the name from the uploaded folder structure
        cb(null, file.fieldname + '-' + uniqueSuffix + '-' + file.originalname.replace(/[^a-zA-Z0-9_.-]/g, '_'));
    }
});

// --- Use .any() for more robust handling of large mixed uploads ---
const upload = multer({
    storage: storage,
    limits: { 
        files: 1500, 
        fileSize: 1024 * 1024 * 200 // 200MB per file
    }
}).any(); // Use .any()
console.log("SERVER.JS: Multer is using .any() to handle all incoming files.");
// --- END ---

// --- Middleware ---
// Serve static files (HTML, CSS, JS) from the parent directory
app.use(express.static(path.join(__dirname, '..'))); // Serve files from 'animal-detection-app'

// --- API Endpoint ---
app.post('/run-model', (req, res) => {
    console.log("SERVER.JS: /run-model route hit. Request Headers:", JSON.stringify(req.headers, null, 2));
    console.log("SERVER.JS: Attempting to process upload with .any().");

    upload(req, res, async function (err) {
        if (err) {
            console.log("SERVER.JS: Multer callback - ERROR received (with .any()):", err);
            console.error("Full error object during upload processing (Multer stage, .any()):", err);
            // err.code will be set for MulterErrors (e.g., 'LIMIT_FILE_SIZE')
            if (err instanceof multer.MulterError) {
                return res.status(400).json({ error: "File upload error: " + err.message + (err.field ? ` (field: ${err.field})` : '') });
            }
            return res.status(500).json({ error: "An unexpected error occurred during initial file upload parsing." });
        }

        // With .any(), req.files is an ARRAY of all files.
        // req.body will contain non-file fields.
        console.log("SERVER.JS: Multer callback (.any()) - NO immediate error.");
        console.log("SERVER.JS: Multer callback (.any()) - req.files (array):", req.files ? req.files.map(f => ({ fieldname: f.fieldname, originalname: f.originalname, size: f.size, path: f.path })) : "No files");
        console.log("SERVER.JS: Multer callback (.any()) - req.body:", req.body);

        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded.' });
        }

        // Find the CSV file
        const csvFileObject = req.files.find(file => file.fieldname === 'csvFile');
        if (!csvFileObject) {
            // Cleanup any files already uploaded by .any() if CSV is missing
            req.files.forEach(file => {
                if (file.path && fs.existsSync(file.path)) {
                    fs.unlink(file.path, delErr => { if(delErr) console.error(`Error deleting temp file ${file.path} after missing CSV:`, delErr);});
                }
            });
            return res.status(400).json({ error: 'No CSV file uploaded or it was not named "csvFile".' });
        }
        const inputCsvPath = csvFileObject.path;
        console.log(`SERVER.JS: CSV file found: ${inputCsvPath}`);

        const action = req.body.action;
        let scriptPath, outputDir, outputFileNamePrefix, args, methodName, outputFilePath;
        let photoLocationPaths = {};

        const runId = `run_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
        const runSpecificUploadDir = path.join(baseUploadDir, runId);

        if (action === 'cluster') {
            console.log("SERVER.JS: Action is 'cluster'. Preparing photo directories (processing .any() results).");
            try {
                if (!fs.existsSync(runSpecificUploadDir)) {
                    fs.mkdirSync(runSpecificUploadDir, { recursive: true });
                }

                const locationPhotoFieldsMap = {
                    'photoN01_files': 'N01',
                    'photoN02_files': 'N02',
                    'photoN03_files': 'N03',
                    'photoT01_files': 'T01',
                    'photoT02_files': 'T02',
                    'photoT03_files': 'T03'
                };
                // photoLocationPaths was defined outside, reset it here for current request
                photoLocationPaths = {};

                const photoFiles = req.files.filter(file => file.fieldname !== 'csvFile');

                photoFiles.forEach(file => {
                    const locationCode = locationPhotoFieldsMap[file.fieldname];
                    if (locationCode) {
                        const targetDir = path.join(runSpecificUploadDir, locationCode);
                        if (!photoLocationPaths[locationCode]) {
                            fs.mkdirSync(targetDir, { recursive: true });
                            photoLocationPaths[locationCode] = targetDir;
                        }
                        const newFilePath = path.join(targetDir, file.originalname);
                        // console.log(`SERVER.JS: Processing photo for field ${file.fieldname} (location ${locationCode}). File: ${file.originalname}`);
                        // console.log(`SERVER.JS: Attempting to move ${file.path} to ${newFilePath}`);
                        try {
                            fs.renameSync(file.path, newFilePath);
                            // console.log(`SERVER.JS: Successfully moved ${file.originalname} to ${newFilePath}`);
                        } catch (moveError) {
                            console.error(`SERVER.JS_ERROR: Failed to move ${file.originalname} from ${file.path} to ${newFilePath}`, moveError);
                            if (fs.existsSync(file.path)) {
                                fs.unlinkSync(file.path);
                            }
                        }
                    } else {
                        // console.warn(`SERVER.JS: File with unmapped fieldname ${file.fieldname} found: ${file.originalname}. Deleting from temp.`);
                         if (file.path && fs.existsSync(file.path)) {
                            fs.unlinkSync(file.path);
                        }
                    }
                });
                console.log("SERVER.JS: Finished processing photo files. photoLocationPaths:", photoLocationPaths);

            } catch (dirOrFileError) {
                console.error("SERVER.JS: CRITICAL ERROR during photo directory setup or file processing loop:", dirOrFileError);
                req.files.forEach(file => {
                    if (file.path && fs.existsSync(file.path)) {
                        try { fs.unlinkSync(file.path); } catch (e) { console.error('Error deleting temp file on dir error:', e); }
                    }
                });
                return res.status(500).json({ error: 'Error setting up directories for photo processing.' });
            }

            // --- Setup for dbscan_script.py ---
            scriptPath = path.join(__dirname, 'dbscan_script.py');
            outputDir = clusteredOutputDir;
            outputFileNamePrefix = 'clustered';
            methodName = 'DBSCAN Clustering';

            const epsValue = req.body.epsLevel === 'low' ? "0.5" : 
                             req.body.epsLevel === 'high' ? "20" : "10"; // Default to medium (10)

            const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E5)}`;
            const outputFileName = `${outputFileNamePrefix}-${uniqueSuffix}.csv`; // Generate unique name
            outputFilePath = path.join(outputDir, outputFileName);
            const photoLocationPathsJson = JSON.stringify(photoLocationPaths);

            // Ensure inputCsvPath is defined and used correctly from the multer processing step
            // It should have been set earlier in the /run-model handler
            if (!inputCsvPath || !fs.existsSync(inputCsvPath)) {
                console.error("SERVER.JS_ERROR: inputCsvPath is not defined or file does not exist for clustering.");
                // Clean up temporary files if any were created for this failed request
                if (req.files && req.files.length > 0) {
                    req.files.forEach(file => {
                        if (file.path && fs.existsSync(file.path)) {
                            try { fs.unlinkSync(file.path); } catch (e) { console.error('Error deleting temp file on inputCsvPath error:', e); }
                        }
                    });
                }
                // Also clean up runSpecificUploadDir if it was created and is empty
                if (fs.existsSync(runSpecificUploadDir) && fs.readdirSync(runSpecificUploadDir).length === 0) {
                    try { fs.rmdirSync(runSpecificUploadDir); } catch(e) { console.error('Error deleting empty runSpecificUploadDir:', e); }
                }
                return res.status(500).json({ error: 'Critical error: Input CSV path not available for clustering.' });
            }

            args = [
                inputCsvPath,       // Correct: Path to the uploaded CSV
                outputFilePath,     // Correct: Path for the output clustered CSV
                epsValue,           // Correct: EPS value
                photoLocationPathsJson // Correct: JSON string of photo paths map
            ];
            
            // The command is 'python', scriptPath is the first argument to python
            executable = 'python'; 
            // Prepend scriptPath to args for execFile if using it directly,
            // or ensure it's the first element after 'python' if using spawn's array style.
            // For spawn, the scriptPath is the first element of the second argument (the array of args).
            // So, the args array for spawn should be: [scriptPath, inputCsvPath, outputFilePath, ...]

            // Corrected for spawn:
            const spawnArgs = [
                scriptPath,
                inputCsvPath,
                outputFilePath,
                epsValue,
                photoLocationPathsJson
            ];

            console.log(`Executing: "${executable}" "${spawnArgs.join('" "')}"`); // For logging

            // --- End of setup for dbscan_script.py ---
            scriptToRun = spawn(executable, spawnArgs);

        } else if (action === 'filter') {
            methodName = 'Time-based filtering';
            scriptPath = path.join(__dirname, 'filter_script.R');
            outputDir = filteredOutputDir;
            const filterMinutes = req.body.filterMinutes || '30';
            
            if (!filterMinutes || isNaN(parseInt(filterMinutes)) || parseInt(filterMinutes) < 1) {
                fs.unlink(inputCsvPath, (delErr) => { if (delErr) console.error("Error deleting temp input file (invalid filterMinutes):", delErr); });
                return res.status(400).json({ error: 'Invalid Filter Minutes value provided. Must be a number greater than 0.' });
            }

            outputFileNamePrefix = `filtered-${filterMinutes}min-`;
            args = [inputCsvPath, 'PLACEHOLDER_FOR_OUTPUT_FILE', filterMinutes];
        } else {
            fs.unlink(inputCsvPath, (delErr) => { if (delErr) console.error("Error deleting temp input file (unknown action):", delErr); });
            if (Object.keys(photoLocationPaths).length > 0) { // Cleanup runSpecificUploadDir if created
                 fs.rm(runSpecificUploadDir, { recursive: true, force: true }, (rmErr) => {
                    if (rmErr) console.error(`Error cleaning up ${runSpecificUploadDir}:`, rmErr);
                });
            }
            return res.status(400).json({ error: 'Invalid action specified.' });
        }

        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E5)}`;
        const outputFileName = `${outputFileNamePrefix}${uniqueSuffix}.csv`;
        outputFilePath = path.join(outputDir, outputFileName);
        
        args[1] = outputFilePath; // Set the actual output file path

        const command = action === 'filter' ? (process.env.RSCRIPT_PATH || 'Rscript') : (process.env.PYTHON_PATH || 'python');
        const scriptArgs = [scriptPath, ...args];

        console.log(`Executing: "${command}" "${scriptArgs.join('" "')}"`);

        const pythonProcess = spawn(command, scriptArgs); // Ensure 'python' is correct

        let scriptStdout = '';
        let scriptStderr = '';

        console.log(`SERVER.JS: Spawning Python script: ${scriptArgs.join(' ')}`);

        pythonProcess.stdout.on('data', (data) => {
            const dataStr = data.toString();
            scriptStdout += dataStr;
            // Log the raw chunks to see if anything comes through at all
            console.log(`PYTHON_STDOUT_CHUNK: ${dataStr}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            const dataStr = data.toString();
            scriptStderr += dataStr;
            // Log the raw chunks
            console.error(`PYTHON_STDERR_CHUNK: ${dataStr}`);
        });

        pythonProcess.on('error', (err) => {
            // This event is emitted if the process could not be spawned,
            // or could not be killed, or sending a message to the child process failed.
            console.error('SERVER.JS_ERROR: Failed to start or interact with Python subprocess.', err);
            // Make sure to send an error response to the client
            if (!res.headersSent) {
                res.status(500).json({ error: 'Failed to start Python script.', details: err.message });
            }
        });

        pythonProcess.on('close', (code, signal) => {
            console.log(`SERVER.JS: Python script process closed with code ${code} and signal ${signal}`);
            console.log("SERVER.JS: --- Full Python stdout ---");
            console.log(scriptStdout);
            console.log("SERVER.JS: --- Full Python stderr ---");
            console.error(scriptStderr);

            if (code !== 0) {
                console.error(`SERVER.JS_ERROR: Python script exited with error code ${code}.`);
                if (!res.headersSent) {
                    try {
                        const errorJson = JSON.parse(scriptStderr || scriptStdout);
                        res.status(500).json({ error: 'Python script error', details: errorJson, stderr: scriptStderr, stdout: scriptStdout });
                    } catch (e) {
                        res.status(500).json({ error: `Python script failed with code ${code}.`, stderr: scriptStderr, stdout: scriptStdout });
                    }
                }
            } else { // Python script exited with code 0 (success)
                if (!res.headersSent) {
                    // For other scripts (like dbscan_script.py or filter_script.py), expect JSON
                    try {
                        const results = JSON.parse(scriptStdout);
                        let responseJson = { ...results };

                        if (action === 'cluster') {
                            if (results.clusters && results.output_file) {
                                const resultsId = crypto.randomBytes(16).toString('hex');
                                clusterResultsStore[resultsId] = {
                                    ...results,
                                    outputCsvPath: outputFilePath, // outputFilePath is defined in the outer scope
                                    timestamp: Date.now()
                                };
                                setTimeout(() => delete clusterResultsStore[resultsId], RESULT_EXPIRY_TIME);
                                console.log(`SERVER.JS: Stored cluster results with ID: ${resultsId}`);
                                responseJson.resultsId = resultsId;
                                responseJson.outputFileName = path.basename(outputFilePath);
                            } else {
                                console.error("SERVER.JS_ERROR: Cluster script output valid JSON, but missing expected fields (clusters, output_file).", results);
                                return res.status(500).json({ error: "Cluster script produced JSON, but content is not as expected.", details: scriptStdout });
                            }
                        } else if (action === 'filter') {
                            if (results.output_file) {
                                responseJson.outputFileName = path.basename(outputFilePath);
                            } else {
                                console.error("SERVER.JS_ERROR: Filter script output valid JSON, but missing expected field (output_file).", results);
                                return res.status(500).json({ error: "Filter script produced JSON, but content is not as expected.", details: scriptStdout });
                            }
                        }
                        res.json(responseJson);
                    } catch (e) {
                        console.error("SERVER.JS_ERROR: Could not parse Python script output as JSON for action:", action, e);
                        res.status(500).json({ error: "Failed to parse script output as JSON.", details: scriptStdout });
                    }
                }
            }
            // Cleanup: Delete the run-specific photo directory and the input CSV
            if (fs.existsSync(runSpecificUploadDir)) {
                fs.rm(runSpecificUploadDir, { recursive: true, force: true }, (rmErr) => {
                    if (rmErr) console.error(`Error cleaning up ${runSpecificUploadDir}:`, rmErr);
                });
            }
            if (inputCsvPath && fs.existsSync(inputCsvPath)) {
                fs.unlink(inputCsvPath, (delErr) => { if (delErr) console.error("Error deleting temp input CSV after script execution:", delErr); });
            }
        });
    });
});

// --- New API Endpoint to fetch stored cluster details ---
app.get('/api/cluster-details/:resultsId', (req, res) => {
    const resultsId = req.params.resultsId;
    const details = clusterResultsStore[resultsId];

    if (details) {
        res.json(details);
    } else {
        res.status(404).json({ error: 'Cluster results not found or expired.' });
    }
});

// --- New: API Endpoint for Downloading Filtered Files ---
app.get('/download-filtered/:fileId', (req, res) => {
    const fileId = req.params.fileId;

    // Basic validation/sanitization of filename (prevent directory traversal)
    if (!fileId || fileId.includes('..') || fileId.includes('/') || fileId.includes('\\')) {
        return res.status(400).send('Invalid file ID.');
    }

    const filePath = path.join(filteredOutputDir, fileId);

    // Check if file exists
    if (fs.existsSync(filePath)) {
        // Extract user-friendly name (e.g., filtered_30min_data.csv)
        const downloadName = fileId.replace(/-\d+\.csv$/, '_data.csv').replace(/^filtered-/, 'filtered_');
        // Use res.download to send the file
        // It sets Content-Disposition header for download prompt
        // Optional: provide a user-friendly download name as the second argument
        res.download(filePath, downloadName, (err) => {
            if (err) {
                // Handle errors that occur during streaming the download
                console.error("Error sending file:", err);
                // Avoid sending another response if headers were already sent
                if (!res.headersSent) {
                    res.status(500).send('Error downloading the file.');
                }
            }
            // Note: We are NOT deleting the file after download here,
            // implement cleanup logic separately if needed.
        });
    } else {
        // File not found
        res.status(404).send('File not found.');
    }
});

// --- New: Download Endpoint for Clustered Files ---
app.get('/download-clustered/:fileId', (req, res) => {
    const fileId = req.params.fileId;
    // Basic validation/sanitization
    if (!fileId || fileId.includes('..') || fileId.includes('/') || fileId.includes('\\')) {
        return res.status(400).send('Invalid file ID.');
    }
    const filePath = path.join(clusteredOutputDir, fileId); // Look in clustered dir

    if (fs.existsSync(filePath)) {
        // Provide a user-friendly download name
        const downloadName = fileId.replace(/-\d+\.csv$/, '.csv').replace(/^clustered-/, 'clustered_');
        res.download(filePath, downloadName, (err) => {
            if (err) {
                console.error("Error sending clustered file:", err);
                if (!res.headersSent) {
                    res.status(500).send('Error downloading the file.');
                }
            }
            // Optional: Cleanup downloaded file after sending?
            // fs.unlink(filePath, (delErr) => { if (delErr) console.error("Error deleting downloaded clustered file:", delErr); });
        });
    } else {
        res.status(404).send('Clustered file not found.');
    }
});

// --- Endpoint to fetch stored cluster results ---
app.get('/results/:resultsId', (req, res) => {
    const resultsId = req.params.resultsId;
    // Log the received ID and the current state of the store for debugging
    console.log(`SERVER.JS: /results/:resultsId endpoint hit. Requested ID: "${resultsId}"`);
    console.log(`SERVER.JS: Current clusterResultsStore keys: [${Object.keys(clusterResultsStore).join(', ')}]`);
    
    const results = clusterResultsStore[resultsId];
    if (results) {
        console.log(`SERVER.JS: Found results for ID: "${resultsId}"`);
        res.json(results); // Send the full stored JSON data
    } else {
        console.warn(`SERVER.JS: Results NOT FOUND for ID: "${resultsId}"`);
        res.status(404).json({ error: 'Results not found for ID: ' + resultsId });
    }
});

// --- Start Server ---
app.listen(port, () => {
    console.log(`Animal Detection Backend listening at http://localhost:${port}`);
    console.log(`Serving frontend from: ${path.join(__dirname, '..')}`);
}); 