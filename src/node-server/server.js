// server.js

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const PORT = 3000; // You can choose any port that is not in use

app.use(cors());
app.use(bodyParser.json());

// Define the endpoint for asking questions
app.post('/ask', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:5000/ask', {
            prompt: req.body.prompt
        });
        console.log('Prompt forwarded to flask "/ask" route')
        res.json(response.data);
    } catch (error) {
        console.error('Error communicating with Flask server:', error);
        res.status(500).json({ error: 'Error communicating with Flask server' });
    }
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

app.listen(PORT, () => {
    console.log(`Node.js server is running on http://localhost:${PORT}`);
});
