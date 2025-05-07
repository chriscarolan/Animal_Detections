const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process'); // Use execFile for better security/control
const path = require('path');
const fs = require('fs'); // File system module
const crypto = require('crypto'); // For generating unique IDs

const app = express();
const port = 3000; // Port the server will listen on

// --- In-memory store for cluster results ---
// Simple store: object mapping resultId to data.
// In a production app, you'd use a database or a more robust caching solution.
const clusterResultsStore = {};
const RESULT_EXPIRY_TIME = 15 * 60 * 1000; // 15 minutes in milliseconds

// --- File Upload Setup (using Multer) ---
// Configure where to store uploaded files temporarily
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)){
    fs.mkdirSync(uploadDir); // Create 'uploads' directory if it doesn't exist
}
// --- New: Directory for Filtered Outputs ---
const filteredOutputDir = path.join(__dirname, 'filtered_outputs');
if (!fs.existsSync(filteredOutputDir)){
    fs.mkdirSync(filteredOutputDir); // Create directory for results
}
// New: Directory for Clustered Outputs
const clusteredOutputDir = path.join(__dirname, 'clustered_outputs');
if (!fs.existsSync(clusteredOutputDir)){
    fs.mkdirSync(clusteredOutputDir); // Create directory for clustered outputs
}
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir); // Save to 'uploads' directory
    },
    filename: function (req, file, cb) {
        // Create a unique filename to avoid conflicts
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

// Configure Multer to handle form-data
const upload = multer({ storage: storage }).fields([
    { name: 'csvFile', maxCount: 1 },
    { name: 'action', maxCount: 1 },
    { name: 'filterMinutes', maxCount: 1 } // filterMinutes is expected
]);

// --- Middleware ---
// Serve static files (HTML, CSS, JS) from the parent directory
app.use(express.static(path.join(__dirname, '..'))); // Serve files from 'animal-detection-app'

// --- API Endpoint ---
app.post('/run-model', (req, res) => {
    upload(req, res, function (err) {
        if (err instanceof multer.MulterError) {
            console.error("Multer error:", err);
            return res.status(500).json({ error: "File upload error: " + err.message });
        } else if (err) {
            console.error("Unknown upload error:", err);
            return res.status(500).json({ error: "An unknown error occurred during file upload." });
        }

        if (!req.files || !req.files.csvFile || !req.files.csvFile[0]) {
            return res.status(400).json({ error: 'No CSV file uploaded.' });
        }
        if (!req.body.action) {
            return res.status(400).json({ error: 'No action specified.' });
        }

        const action = req.body.action;
        const inputFilePath = req.files.csvFile[0].path;
        let scriptPath, outputDir, outputFileNamePrefix, args, methodName;

        if (action === 'filter') {
            methodName = 'Time-based filtering';
            scriptPath = path.join(__dirname, 'filter_script.R');
            outputDir = filteredOutputDir;
            const filterMinutes = req.body.filterMinutes || '30'; // Default to 30 if not provided
            
            // Validate filterMinutes
            if (!filterMinutes || isNaN(parseInt(filterMinutes)) || parseInt(filterMinutes) < 1) {
                fs.unlink(inputFilePath, (delErr) => { if (delErr) console.error("Error deleting temp input file (invalid filterMinutes):", delErr); });
                return res.status(400).json({ error: 'Invalid Filter Minutes value provided. Must be a number greater than 0.' });
            }

            outputFileNamePrefix = `filtered-${filterMinutes}min-`;
            // Ensure args has three elements for the R script: input, output_placeholder, filterMinutes
            args = [inputFilePath, 'PLACEHOLDER_FOR_OUTPUT_FILE', filterMinutes]; 
            // The 'PLACEHOLDER_FOR_OUTPUT_FILE' will be replaced by the actual outputFilePath later
        } else if (action === 'cluster') {
            methodName = 'DBSCAN clustering';
            scriptPath = path.join(__dirname, 'dbscan_script.py');
            outputDir = clusteredOutputDir;
            outputFileNamePrefix = 'clustered-';
            // Python script expects: input_file, output_file
            args = [inputFilePath, 'PLACEHOLDER_FOR_OUTPUT_FILE']; 
        } else {
            fs.unlink(inputFilePath, (delErr) => { if (delErr) console.error("Error deleting temp input file (unknown action):", delErr); });
            return res.status(400).json({ error: 'Invalid action specified.' });
        }

        // Create unique suffix for output file
        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E5)}`;
        const outputFileName = `${outputFileNamePrefix}${uniqueSuffix}.csv`;
        const outputFilePath = path.join(outputDir, outputFileName);
        
        // Now, correctly set the output file path in the args array
        // For filter, it's the second argument (index 1)
        // For cluster, it's also the second argument (index 1)
        args[1] = outputFilePath;

        const command = action === 'filter' ? (process.env.RSCRIPT_PATH || 'Rscript') : (process.env.PYTHON_PATH || 'python');
        const scriptArgs = [scriptPath, ...args];

        console.log(`Executing: "${command}" "${scriptArgs.join('" "')}"`); // Log the exact command

        execFile(command, scriptArgs, (error, stdout, stderr) => {
            fs.unlink(inputFilePath, (err) => { if (err) console.error("Error deleting temp input file after script execution:", err); });

            if (error) {
                console.error(`Script execution error for ${action}:`, error);
                console.error(`Script stderr for ${action}:`, stderr);

                if (outputFilePath && fs.existsSync(outputFilePath)) {
                   fs.unlink(outputFilePath, (delErr) => { if (delErr) console.error(`Error deleting temp output file ${outputFilePath} after script error:`, delErr); });
                }

                // Construct a more controlled error message for the client
                let clientErrorMessage = `Script execution failed.`;
                if (error.code) {
                    clientErrorMessage += ` Exit code: ${error.code}.`;
                }
                // Add a snippet of stderr if it exists, otherwise use the error.message from execFile
                if (stderr && stderr.trim().length > 0) {
                    clientErrorMessage += ` Details: ${stderr.trim().substring(0, 500)}`; // Send first 500 chars of stderr
                } else if (error.message) {
                    clientErrorMessage += ` Message: ${error.message}`;
                }
                // The full details are in the Node.js console.
                return res.status(500).json({ error: clientErrorMessage });
            }

            if (stderr) { // Python might print warnings to stderr but still succeed (exit code 0)
                console.warn(`Script stderr (warnings) for ${action}:`, stderr);
            }

            console.log(`Script stdout for ${action}:`, stdout);

            if (action === 'filter') {
                try {
                    const rOutput = JSON.parse(stdout); // Parse JSON from R
                    const filteredCount = parseInt(rOutput.filtered_count, 10);
                    const totalRecords = parseInt(rOutput.total_records, 10);

                    if (isNaN(filteredCount) || isNaN(totalRecords)) {
                        if (outputFilePath && fs.existsSync(outputFilePath)) fs.unlink(outputFilePath, (delErr) => { if (delErr) console.error("Error deleting temp output file (filter) due to invalid count:", delErr); });
                        console.error("R script output was not a valid JSON with numbers:", stdout);
                        return res.status(500).json({ error: 'R script did not return valid counts.' });
                    }
                    res.json({
                        count: filteredCount,
                        totalRecords: totalRecords, // Pass totalRecords
                        method: methodName,
                        fileId: outputFileName
                    });
                } catch (parseError) {
                    console.error("Error parsing R script JSON output:", parseError, "Output was:", stdout);
                    if (outputFilePath && fs.existsSync(outputFilePath)) fs.unlink(outputFilePath, (delErr) => { if (delErr) console.error("Error deleting temp output file (filter) after R parse error:", delErr); });
                    return res.status(500).json({ error: 'Failed to parse R script output.' });
                }
            } else if (action === 'cluster') {
                try {
                    const clusterDetails = JSON.parse(stdout); // This is the large detailed object
                    const resultsId = crypto.randomBytes(16).toString('hex'); // Generate a unique ID

                    // Store the results
                    clusterResultsStore[resultsId] = clusterDetails;

                    // Set a timeout to delete the stored result after expiry time
                    setTimeout(() => {
                        delete clusterResultsStore[resultsId];
                        console.log(`Expired and deleted cluster result: ${resultsId}`);
                    }, RESULT_EXPIRY_TIME);

                    // Respond with the ID and the fileId for download
                    res.json({
                        method: methodName,
                        resultsId: resultsId, // Send this ID to the client
                        fileId: outputFileName  // For downloading the CSV
                    });
                } catch (parseError) {
                    console.error("Error parsing Python script JSON output:", parseError, "Output was:", stdout);
                    // If parsing fails, still try to delete the output file as it might be corrupt/empty
                    if (outputFilePath && fs.existsSync(outputFilePath)) fs.unlink(outputFilePath, (delErr) => { if (delErr) console.error("Error deleting temp output file (cluster) after parse error:", delErr); });
                    return res.status(500).json({ error: 'Failed to parse clustering results from script.' });
                }
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

// --- Start Server ---
app.listen(port, () => {
    console.log(`Animal Detection Backend listening at http://localhost:${port}`);
    console.log(`Serving frontend from: ${path.join(__dirname, '..')}`);
}); 