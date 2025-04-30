const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process'); // Use execFile for better security/control
const path = require('path');
const fs = require('fs'); // File system module

const app = express();
const port = 3000; // Port the server will listen on

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
const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // Limit file size (e.g., 10MB)
    fileFilter: function (req, file, cb) {
        // Accept only CSV files
        if (path.extname(file.originalname).toLowerCase() !== '.csv') {
            return cb(new Error('Only .csv files are allowed!'), false);
        }
        cb(null, true);
    }
}).single('csvFile'); // Matches the name attribute in the frontend form/FormData

// --- Middleware ---
// Serve static files (HTML, CSS, JS) from the parent directory
app.use(express.static(path.join(__dirname, '..'))); // Serve files from 'animal-detection-app'

// --- API Endpoint ---
app.post('/run-model', (req, res) => {
    upload(req, res, (uploadError) => {
        if (uploadError) {
            console.error("Upload Error:", uploadError);
            // Handle Multer errors (e.g., file size limit, wrong file type)
            return res.status(400).json({ error: `File upload failed: ${uploadError.message}` });
        }

        if (!req.file) {
            return res.status(400).json({ error: 'No CSV file uploaded.' });
        }
        if (!req.body.action) {
            return res.status(400).json({ error: 'No action specified (filter/cluster).' });
        }

        const action = req.body.action;
        const inputFilePath = req.file.path; // Path to the uploaded file
        let scriptPath = '';
        let methodName = '';
        let outputFileName = ''; // To store the name of the generated filtered file
        let outputFilePath = ''; // Full path for the R script output
        let scriptArgs = []; // Array to hold arguments for R script

        // --- Select Script based on Action ---
        if (action === 'filter') {
            // New: Get filter minutes from request body
            const filterMinutes = req.body.filterMinutes;
            if (!filterMinutes || isNaN(parseInt(filterMinutes, 10)) || parseInt(filterMinutes, 10) <= 0) {
                fs.unlink(inputFilePath, (err) => { if (err) console.error("Error deleting temp input file:", err); });
                return res.status(400).json({ error: 'Invalid or missing filter minutes value provided.' });
            }

            scriptPath = path.join(__dirname, 'filter_script.R');
            methodName = `filtering algorithm (${filterMinutes} min)`; // Update method name
            outputFileName = `filtered-${filterMinutes}min-${Date.now()}.csv`; // Include minutes in filename
            outputFilePath = path.join(filteredOutputDir, outputFileName);

            // Set arguments for R script: script path, input csv, output csv, minutes
            scriptArgs = [scriptPath, inputFilePath, outputFilePath, filterMinutes];
        } else if (action === 'cluster') {
            // scriptPath = path.join(__dirname, 'dbscan_script.R'); // ** TODO: Create this script **
            // methodName = 'DBSCAN model';
            // For now, return an error or simulate if cluster isn't implemented
             // Clean up uploaded file
            fs.unlink(inputFilePath, (err) => { if (err) console.error("Error deleting temp input file:", err); });
            return res.status(501).json({ error: 'Cluster action not yet implemented.' });
        } else {
             // Clean up uploaded file
            fs.unlink(inputFilePath, (err) => { if (err) console.error("Error deleting temp input file:", err); });
            return res.status(400).json({ error: 'Invalid action specified.' });
        }

         // --- Execute the R Script ---
         // ** MODIFY THIS LINE **
         // Replace 'Rscript' with the full, quoted path to your Rscript.exe
         // Use the user's actual path here:
         const rscriptExecutable = "C:\\Program Files\\R\\R-4.1.1\\bin\\Rscript.exe"; // <--- YOUR ACTUAL PATH

         // Pass the arguments array to execFile
         console.log(`Executing: "${rscriptExecutable}"`, scriptArgs.map(arg => `"${arg}"`).join(' ')); // Log execution clearly
         execFile(rscriptExecutable, scriptArgs, (error, stdout, stderr) => {
             // Clean up the *uploaded input* file regardless of R script outcome
             fs.unlink(inputFilePath, (err) => {
                 if (err) console.error("Error deleting temp input file:", err);
             });

             if (error) {
                 console.error(`R Script Error: ${error.message}`);
                 console.error(`R Script stderr: ${stderr}`);
                 // If R script failed, attempt to delete the potentially empty/incomplete output file
                 if (outputFilePath && fs.existsSync(outputFilePath)) {
                     fs.unlink(outputFilePath, (delErr) => {
                         if (delErr) console.error("Error deleting temp output file after R error:", delErr);
                     });
                 }
                 // Send a more specific error message if stderr provides useful info
                 const errorMessage = stderr || error.message;
                 return res.status(500).json({ error: `Error running R script: ${errorMessage}` });
             }

             // --- Process R Script Output ---
             const output = stdout.trim();
             console.log(`R Script stdout (count): ${output}`);

             // Validate output (expecting a number)
             const count = parseInt(output, 10);
             if (isNaN(count)) {
                 console.error("R script output was not a valid number:", output);
                  // If count is invalid, delete the generated output file
                 if (outputFilePath && fs.existsSync(outputFilePath)) {
                     fs.unlink(outputFilePath, (delErr) => {
                         if (delErr) console.error("Error deleting temp output file due to invalid count:", delErr);
                     });
                 }
                 return res.status(500).json({ error: 'R script did not return a valid count.' });
             }

             // --- Send Success Response (including the output filename) ---
             res.json({
                 count: count,
                 method: methodName,
                 fileId: outputFileName // Send the generated filename back
             });
         });
    });
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

// --- Start Server ---
app.listen(port, () => {
    console.log(`Animal Detection Backend listening at http://localhost:${port}`);
    console.log(`Serving frontend from: ${path.join(__dirname, '..')}`);
}); 