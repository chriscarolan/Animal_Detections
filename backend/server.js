const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process'); // Use execFile for better security/control
const path = require('path');
const fs = require('fs'); // File system module
const crypto = require('crypto'); // For generating unique IDs
const { spawn } = require('child_process'); // Added for spawn method

const app = express();
const port = 3000; // Port the server will listen on

// Define location IDs globally for reuse
const LOCATION_IDS = ['N01', 'N02', 'N03', 'T01', 'T02', 'T03'];

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
        const csvFile = req.files.find(f => f.fieldname === 'csvFile');
        if (!csvFile) {
            console.error("SERVER.JS_ERROR: CSV file not found in request.");
            return res.status(400).json({ error: "CSV file is required." });
        }
        const csvFilePath = csvFile.path;
        console.log(`SERVER.JS: CSV file found: ${csvFilePath}`);

        const action = req.body.action;

        // Declare variables that will be set based on action
        let scriptPath;
        let outputDir;
        let outputFileNamePrefix;
        let outputFilePath;
        let executable;
        let scriptArgumentsForSpawn;
        let methodName;
        let runSpecificUploadDir = null; // For photo uploads, specific to cluster action

        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E5)}`; // Common unique suffix

        if (action === 'filter') {
            methodName = 'Time-based filtering';
            scriptPath = path.join(__dirname, 'filter_script.R');
            outputDir = filteredOutputDir;
            const filterMinutes = req.body.filterMinutes || '30';
            
            if (!filterMinutes || isNaN(parseInt(filterMinutes)) || parseInt(filterMinutes) < 1) {
                fs.unlink(csvFilePath, (delErr) => { if (delErr) console.error("Error deleting temp input file (invalid filterMinutes):", delErr); });
                return res.status(400).json({ error: 'Invalid Filter Minutes value provided. Must be a number greater than 0.' });
            }

            outputFileNamePrefix = `filtered-${filterMinutes}min-`;
            const outputFileName = `${outputFileNamePrefix}${uniqueSuffix}.csv`;
            outputFilePath = path.join(outputDir, outputFileName);
            executable = process.env.RSCRIPT_PATH || 'Rscript';
            scriptArgumentsForSpawn = [scriptPath, csvFilePath, outputFilePath, filterMinutes];

        } else if (action === 'cluster') {
            methodName = 'DBSCAN Clustering';
            scriptPath = path.join(__dirname, 'dbscan_script.py');
            outputDir = clusteredOutputDir;
            outputFileNamePrefix = "clustered-"; // Define prefix for cluster
            executable = process.env.PYTHON_PATH || 'python';
            
            const epsLevel = req.body.epsLevel || 'medium';
            const epsMapping = {
                "low": "0.1", "medium": "10", "high": "60", "extreme": "200"
            };
            const epsValueForScript = epsMapping[epsLevel] || "10";

            const runId = `${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
            runSpecificUploadDir = path.join(baseUploadDir, `run_${runId}`); // Set for cluster action
            if (!fs.existsSync(runSpecificUploadDir)) {
                fs.mkdirSync(runSpecificUploadDir, { recursive: true });
            }

            const locationProcessingConfigForPython = {};
            for (const locId of LOCATION_IDS) {
                const fieldNameForLoc = `photo${locId}_files`;
                const filesForLoc = req.files.filter(f => f.fieldname === fieldNameForLoc);
                const useImageForLoc = req.body[`useImage${locId}`] === 'yes';

                locationProcessingConfigForPython[locId] = {
                    use_image: useImageForLoc ? "yes" : "no",
                    path: null
                };

                if (useImageForLoc && filesForLoc.length > 0) {
                    const locDir = path.join(runSpecificUploadDir, locId);
                    if (!fs.existsSync(locDir)) {
                        fs.mkdirSync(locDir, { recursive: true });
                    }
                    
                    for (const file of filesForLoc) {
                        if (file.originalname.startsWith('._')) {
                            console.log(`SERVER.JS: Skipping metadata file: ${file.originalname} for ${locId}`);
                            fs.unlink(file.path, err => {
                                if (err) console.error(`SERVER.JS_ERROR: Failed to delete temp metadata file ${file.path}: ${err}`);
                            });
                            continue;
                        }
                        const destPath = path.join(locDir, file.originalname);
                        try {
                            await fs.promises.rename(file.path, destPath);
                        } catch (moveError) {
                            console.error(`SERVER.JS_ERROR: Failed to move file ${file.path} to ${destPath}: ${moveError}. Attempting copy then unlink.`);
                            try {
                                await fs.promises.copyFile(file.path, destPath);
                                await fs.promises.unlink(file.path);
                            } catch (copyUnlinkError) {
                                console.error(`SERVER.JS_ERROR: Critical error during copy/unlink for ${file.path}: ${copyUnlinkError}.`);
                            }
                        }
                    }
                    const filesInLocDir = await fs.promises.readdir(locDir);
                    if (filesInLocDir.length > 0) {
                        locationProcessingConfigForPython[locId].path = locDir;
                    } else {
                        console.log(`SERVER.JS: Photo folder for ${locId} was created but is empty after filtering. No path will be sent to Python.`);
                    }
                } else if (useImageForLoc && filesForLoc.length === 0) {
                    console.log(`SERVER.JS: User selected 'Yes' for image processing for ${locId} but no files were uploaded. Path will be null.`);
                }
            }
            
            console.log("SERVER.JS: Finished processing photo files. locationProcessingConfigForPython:", JSON.stringify(locationProcessingConfigForPython, null, 2));

            const outputFileName = `${outputFileNamePrefix}${uniqueSuffix}.csv`;
            outputFilePath = path.join(outputDir, outputFileName);
            const locationConfigJsonString = JSON.stringify(locationProcessingConfigForPython);

            scriptArgumentsForSpawn = [
                scriptPath,
                csvFilePath,
                outputFilePath,
                epsValueForScript,
                locationConfigJsonString
            ];
        } else {
            fs.unlink(csvFilePath, (delErr) => { if (delErr) console.error("Error deleting temp input file (unknown action):", delErr); });
            // No runSpecificUploadDir to clean up here as it's initialized to null and only set for 'cluster'
            return res.status(400).json({ error: 'Invalid action specified.' });
        }

        console.log(`SERVER.JS: Spawning script. Executable: "${executable}", Args: "${scriptArgumentsForSpawn.join('" "')}"`);
        const childProcess = spawn(executable, scriptArgumentsForSpawn);

        let scriptStdout = '';
        let scriptStderr = '';

        childProcess.stdout.on('data', (data) => {
            const dataStr = data.toString();
            scriptStdout += dataStr;
            console.log(`PYTHON_STDOUT_CHUNK: ${dataStr}`);
        });

        childProcess.stderr.on('data', (data) => {
            const dataStr = data.toString();
            scriptStderr += dataStr;
            console.error(`PYTHON_STDERR_CHUNK: ${dataStr}`);
        });

        childProcess.on('error', (err) => {
            console.error('SERVER.JS_ERROR: Failed to start or interact with child subprocess.', err);
            if (!res.headersSent) {
                res.status(500).json({ error: `Failed to start ${methodName} script.`, details: err.message });
            }
        });

        childProcess.on('close', (code, signal) => {
            console.log(`SERVER.JS: Script process for "${methodName}" closed with code ${code} and signal ${signal}`);
            console.log("SERVER.JS: --- Full Python stdout ---");
            console.log(scriptStdout);
            console.log("SERVER.JS: --- Full Python stderr ---");
            console.error(scriptStderr);

            if (code !== 0) {
                console.error(`SERVER.JS_ERROR: ${methodName} script exited with error code ${code}.`);
                if (!res.headersSent) {
                    res.status(500).json({ 
                        error: `${methodName} script failed.`, 
                        details: `Exit code: ${code}. Check server logs for Python/R stderr.`,
                        stderr: scriptStderr.slice(-1000) // Send last 1000 chars of stderr
                    });
                }
            } else {
                if (!scriptStdout.trim()) {
                    console.error(`SERVER.JS_ERROR: ${methodName} script exited successfully but produced no stdout.`);
                    if (!res.headersSent) {
                        res.status(500).json({ error: `${methodName} script produced no output.` });
                    }
                } else {
                    try {
                        const results = JSON.parse(scriptStdout);
                        let responseJson = { ...results }; // Base response
                        
                        if (action === 'cluster') {
                            if (results.clusters && results.output_file) {
                                const resultId = `res_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
                                clusterResultsStore[resultId] = results; // Store full results
                                responseJson.resultsId = resultId; // Send ID to client
                                responseJson.outputFileName = path.basename(outputFilePath); // Send the generated output file name

                                // Set a timeout to delete the stored result
                                setTimeout(() => {
                                    delete clusterResultsStore[resultId];
                                    console.log(`SERVER.JS: Expired and deleted stored result: ${resultId}`);
                                }, RESULT_EXPIRY_TIME);
                            } else {
                                console.error("SERVER.JS_ERROR: Cluster script output valid JSON, but missing expected fields (clusters or output_file).", results);
                                if (!res.headersSent) {
                                   return res.status(500).json({ error: "Cluster script produced JSON, but content is not as expected.", details: scriptStdout });
                                }
                            }
                        } else if (action === 'filter') {
                            if (results.output_file) {
                                responseJson.outputFileName = path.basename(outputFilePath);
                            } else {
                                console.error("SERVER.JS_ERROR: Filter script output valid JSON, but missing expected field (output_file).", results);
                                if (!res.headersSent) {
                                    return res.status(500).json({ error: "Filter script produced JSON, but content is not as expected.", details: scriptStdout });
                                }
                            }
                        }
                        if (!res.headersSent) {
                            res.json(responseJson);
                        }
                    } catch (e) {
                        console.error(`SERVER.JS_ERROR: Could not parse ${methodName} script output as JSON for action:`, action, e);
                        if (!res.headersSent) {
                            res.status(500).json({ error: `Failed to parse ${methodName} script output as JSON.`, details: scriptStdout });
                        }
                    }
                }
            }
            // Cleanup
            if (runSpecificUploadDir && fs.existsSync(runSpecificUploadDir)) {
                fs.rm(runSpecificUploadDir, { recursive: true, force: true }, (rmErr) => {
                    if (rmErr) console.error(`Error cleaning up ${runSpecificUploadDir}:`, rmErr);
                });
            }
            if (csvFilePath && fs.existsSync(csvFilePath)) {
                fs.unlink(csvFilePath, (delErr) => { if (delErr) console.error("Error deleting temp input CSV after script execution:", delErr); });
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