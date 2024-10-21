const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const { db, closeDb } = require('./database');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

const extractUserId = (req, res, next) => {
    const userId = req.headers['user-id']; 
    req.userId = userId;
    next();
};


app.use(extractUserId);



// Setup multer for file handling
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Route to handle incoming data from the React frontend
app.post('/predict', async (req, res) => {
    try {
        // Send data to the Flask server
        // console.log(req.body)
        const response = await axios.post('http://127.0.0.1:5000/predict', req.body);
        
        // Return the response from Flask to the React frontend
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'An error occurred while processing your request.' });
    }
});

app.post('/upload-image', upload.single('image'), async (req, res) => {

    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
  
    try {
      // Forward the image to Flask server for processing
      const response = await axios.post('http://127.0.0.1:5000/detect', {
        image: req.file.buffer.toString('base64'), 
      });
  
      const userId = req.userId; // Use extracted user ID
      const image = req.file.buffer.toString('base64');
      const createdOn = new Date().toISOString();
      const result = response.data; // Use the result from Flask
  
      const newRetino = {
        id: uuidv4(),
        user_id: userId,
        retinal_scan: image,
        result: JSON.stringify(result), // Ensure the result is a string
        createdOn: createdOn,
      };
  
      // Insert into the database
      db.run('INSERT INTO user_retino (id, user_id, retinal_scan, result, createdOn) VALUES (?, ?, ?, ?, ?)', 
        [newRetino.id, newRetino.user_id, newRetino.retinal_scan, newRetino.result, newRetino.createdOn], function(err) {
          if (err) {
            console.error('Error inserting new user retino data: ' + err.message);
            return res.status(500).json({ error: 'Internal server error' });
          } else {
            console.log(`New Retino record created: ${newRetino.id}`);
            // Send response as a string
            return res.status(201).json(response.data);
          }
      });
  
    } catch (error) {
      console.error('Error communicating with Flask server:', error);
      res.status(500).json({ error: 'Error communicating with Flask server' });
    }
  });  
  


// Endpoint to get API keys for a specific user
app.get('/api-keys', (req, res) => {
    const userId = req.userId; // Use extracted user ID
    db.all('SELECT * FROM api_keys WHERE user_id = ?', [userId], (err, rows) => {
        if (err) {
            console.error('Error fetching API keys: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            res.json(rows);
        }
    });
});

// Endpoint to create a new API key
app.post('/api-keys', (req, res) => {
    const userId = req.userId; // Use extracted user ID
    const { name } = req.body; // Get the name from the request body
    const createdOn = new Date().toISOString(); // Create a timestamp for createdOn
    const expireOn = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(); // Set expiration to 30 days from now
    
    const newKey = {
        id: uuidv4(), // Generate a unique ID for the key
        key: uuidv4(), // Generate a unique API key (UUID)
        name: name, // Associate the name with the key
        user_id: userId, // Associate key with user ID
        createdOn: createdOn,
        expireOn: expireOn
    };
    
    db.run('INSERT INTO api_keys (id, key, user_id, name, createdOn, expireOn) VALUES (?, ?, ?, ?, ?, ?)', 
        [newKey.id, newKey.key, newKey.user_id, newKey.name, newKey.createdOn, newKey.expireOn], function(err) {
        if (err) {
            console.error('Error inserting new API key: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            console.log(`New API key created: ${newKey.key}`); // Log the new API key creation
            res.status(201).json(newKey);
        }
    });
});

// Endpoint to save chat history
app.post('/chat-history', (req, res) => {
    console.log('posting User chat to db')
    const userId = req.userId;
    const { prompt, response } = req.body;
    const createdAt = new Date().toISOString();

    const newChat = {
        id: uuidv4(),
        user_id: userId,
        prompt: prompt,
        response: response,
        createdAt: createdAt,
    };

    db.run('INSERT INTO chat_history (id, user_id, prompt, response, createdAt) VALUES (?, ?, ?, ?, ?)', 
        [newChat.id, newChat.user_id, newChat.prompt, newChat.response, newChat.createdAt], function(err) {
            if (err) {
                console.error('Error saving chat history: ' + err.message);
                res.status(500).json({ error: 'Internal server error' });
            } else {
                res.status(201).json(newChat);
            }
    });
});

// Endpoint to get chat history for a user
app.get('/chat-history', (req, res) => {
    console.log('fetching user chat_hist from db...');
    const userId = req.userId;

    db.all('SELECT * FROM chat_history WHERE user_id = ? ORDER BY createdAt DESC', [userId], (err, rows) => {
        if (err) {
            console.error('Error fetching chat history: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            // Reverse the order of the rows
            const reversedRows = rows.reverse();
            res.json(reversedRows);
        }
    });
});

app.get('/chat-summary/:userId', async (req, res) => {
    console.log('Fetching user chat history for summarization');
    const userId = req.params.userId;

    try {
        db.all('SELECT prompt, response FROM chat_history WHERE user_id = ?', [userId], async (err, rows) => {
            if (err) {
                console.error('Error fetching chat history: ' + err.message);
                return res.status(500).json({ error: 'Internal server error' });
            }

            if (!rows || rows.length === 0) {
                console.log('No chat history found for user:', userId);
                return res.status(404).json({ message: 'No chat history found' });
            }

            // Check if chat history contains less than 10 entries (prompts + responses)
            if (rows.length < 10) {
                console.log('Chat history is not long enough to form a conclusive report.');
                return res.status(400).json({ message: 'Chat history is not long enough to form a conclusive report' });
            }

            // Concatenate all prompts and responses into a single string
            const chatHistory = rows.map(row => `Prompt: ${row.prompt}\nResponse: ${row.response}`).join('\n\n');

            try {
                // Send the chat history to the Flask server for summarization and PDF generation
                const response = await fetch('http://127.0.0.1:5000/chat-summary', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        chat_history: chatHistory,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`Failed to summarize chat history: ${response.statusText}`);
                }

                // Read the PDF Blob from the Flask server response
                const pdfBuffer = await response.arrayBuffer(); // Use arrayBuffer for binary data

                // Set the headers for the response
                res.setHeader('Content-Type', 'application/pdf');
                res.setHeader('Content-Disposition', 'attachment; filename=chat_summary.pdf');

                // Send the PDF buffer as the response
                res.end(Buffer.from(pdfBuffer)); // Convert ArrayBuffer to Node.js Buffer and send it
                console.log('Successfully sent chat summary PDF to client.');
            } catch (fetchError) {
                console.error('Error fetching summary from Flask:', fetchError.message);
                return res.status(500).json({ error: 'Failed to summarize chat history' });
            }
        });
    } catch (err) {
        console.error('Unexpected error:', err.message);
        return res.status(500).json({ error: 'Unexpected server error' });
    }
});



// Endpoint to delete chat history for a user
app.delete('/chat-history', (req, res) => {
    const userId = req.userId;

    db.run('DELETE FROM chat_history WHERE user_id = ?', userId, function(err) {
        if (err) {
            console.error('Error deleting chat history: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else if (this.changes === 0) {
            res.status(404).json({ error: 'No chat history found for this user.' });
        } else {
            console.log(`Chat history deleted for user: ${userId}`); // Log the deletion
            res.status(204).send(); // No content response
        }
    });
});



// Endpoint to delete an API key
app.delete('/api-keys/:id', (req, res) => {
    const keyId = req.params.id;
    
    db.run('DELETE FROM api_keys WHERE id = ?', keyId, function(err) {
        if (err) {
            console.error('Error deleting API key: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else if (this.changes === 0) {
            res.status(404).json({ error: 'API key not found' });
        } else {
            console.log(`API key deleted`);
            res.status(204).send(); 
        }
    });
});

// Define the endpoint for asking questions
app.post('/ask', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/ask', {
            prompt: req.body.prompt
        });
        console.log('Prompt forwarded to Flask "/ask" route');
        res.json(response.data);
    } catch (error) {
        console.error('Error communicating with Flask server:', error);
        res.status(500).json({ error: 'Error communicating with Flask server' });
    }
});

app.post('/api/v1', async (req, res) => {
    console.log('Request body:', req.body); // Log the entire request body
    const { apiKey, procedureType } = req.body;


    if (!apiKey) {
        return res.status(400).json({ error: 'API key is required.' });
    }

    // Check if the API key exists in the database
    db.get('SELECT * FROM api_keys WHERE key = ?', [apiKey], async (err, row) => {
        if (err) {
            console.error('Error validating API key: ' + err.message);
            return res.status(500).json({ error: 'Internal server error' });
        }

        if (!row) {
            return res.status(403).json({ error: 'Invalid API key.' });
        }

        // Check if the API key is expired
        const currentDate = new Date().toISOString();
        if (currentDate > row.expireOn) {
            return res.status(403).json({ error: `API key expired on ${row.expireOn}` });
        }

        // Handle different procedures based on procedureType
        if (procedureType === 'ask') {
            const { prompt } = req.body; // Fetch only the prompt for ask procedure
            if (!prompt) {
                return res.status(400).json({ error: 'Prompt is required for ask procedure.' });
            }
            try {
                const response = await axios.post('http://127.0.0.1:5000/ask', { prompt });
                return res.json(response.data);
            } catch (error) {
                console.error('Error communicating with Flask server:', error);
                return res.status(500).json({ error: 'Error communicating with Flask server' });
            }
        } else if (procedureType === 'predict') {
            // Fetch only the relevant fields for predict procedure
            const { gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level } = req.body;

            // Validate required fields for the predict procedure
            if (!gender || !age || !bmi || !HbA1c_level || !blood_glucose_level) {
                return res.status(400).json({ error: 'Missing required fields for predict procedure.' });
            }

            // Add checks for hypertension and heart disease if they are expected
            if (hypertension === undefined || heart_disease === undefined) {
                return res.status(400).json({ error: 'Hypertension and heart disease fields are required.' });
            }

            try {
                // Forward the diabetes prediction data to the appropriate Flask endpoint
                const diabetesPredictionResponse = await axios.post('http://127.0.0.1:5000/predict', {
                    gender,
                    age,
                    hypertension: hypertension === 'Yes' ? 1 : 0, // Convert Yes/No to 1/0
                    heart_disease: heart_disease === 'Yes' ? 1 : 0, // Convert Yes/No to 1/0
                    smoking_history,
                    bmi,
                    HbA1c_level,
                    blood_glucose_level
                });
                return res.json(diabetesPredictionResponse.data);
            } catch (error) {
                console.error('Error communicating with Flask server for diabetes prediction:', error);
                return res.status(500).json({ error: 'Error communicating with Flask server for diabetes prediction' });
            }
        } else {
            return res.status(400).json({ error: 'Invalid procedure type specified.' });
        }
    });
});

app.post('/update-context', async (req, res) => {
    const { bloodSugar, heartRate, age, glucoseLevels, hb1ac } = req.body;
  
    try {
      // Forward the data to Flask server for processing
      const response = await axios.post('http://127.0.0.1:5000/update-context', req.body);
  
      // Assuming `userId` is sent from the frontend or stored in headers/session
      const userId = req.get('user-id'); // Get user ID from headers or however you track user sessions
  
      const query = `
        INSERT INTO user_health_context (user_id, blood_sugar, heart_rate, age, glucose_levels, hb1ac, updated_on)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          blood_sugar = excluded.blood_sugar,
          heart_rate = excluded.heart_rate,
          age = excluded.age,
          glucose_levels = excluded.glucose_levels,
          hb1ac = excluded.hb1ac,
          updated_on = excluded.updated_on
      `;
  
      const updatedOn = new Date().toISOString(); // Use current timestamp
  
      // Execute the query with the provided values
      db.run(query, [userId, bloodSugar, heartRate, age, glucoseLevels, hb1ac, updatedOn], function (err) {
        if (err) {
          console.error('Error updating user context data:', err.message);
          return res.status(500).json({ error: 'Internal server error' });
        }
  
        console.log(`User context updated in database`);
        res.status(200).json({
          message: 'Context updated successfully',
          result: response.data, // Forward Flask server's response
        });
      });
    } catch (error) {
      console.error('Error communicating with Flask server:', error);
      res.status(500).json({ error: 'Error communicating with Flask server' });
    }
  });
  

// Close database connection when the application exits
process.on('exit', closeDb);

app.listen(PORT, () => {
    console.log(`Node.js server is running on http://localhost:${PORT}`);
});
