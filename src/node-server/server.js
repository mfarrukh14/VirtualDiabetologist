const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const { db, closeDb } = require('./database');

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
    console.log('fetching user chat_hist from db...')
    const userId = req.userId;

    db.all('SELECT * FROM chat_history WHERE user_id = ? ORDER BY createdAt DESC', [userId], (err, rows) => {
        if (err) {
            console.error('Error fetching chat history: ' + err.message);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            res.json(rows);
        }
    });
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
            console.log(`API key deleted: ${keyId}`); // Log the deleted API key
            res.status(204).send(); // No content response
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
    const { apiKey, prompt } = req.body;
    console.log('Received API Key:', apiKey);
    console.log('Received Prompt:', prompt);

    if (!apiKey || !prompt) {
        return res.status(400).json({ error: 'API key and prompt are required.' });
    }

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

        // Forward the prompt to the Flask server
        try {
            const response = await axios.post('http://127.0.0.1:5000/ask', {
                prompt: prompt
            });
            return res.json(response.data);
        } catch (error) {
            console.error('Error communicating with Flask server:', error);
            return res.status(500).json({ error: 'Error communicating with Flask server' });
        }
    });
});



// Define the endpoint for updating context
app.post('/update-context', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/update-context', req.body);
        res.json(response.data);
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
