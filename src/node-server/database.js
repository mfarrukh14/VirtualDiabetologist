// database.js
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Initialize the database
const db = new sqlite3.Database(path.join(__dirname, 'api_keys.db'), (err) => {
    if (err) {
        console.error('Error opening database ' + err.message);
    } else {
        // Create the table if it doesn't exist
        db.run(`CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            user_id TEXT NOT NULL
        )`, (err) => {
            if (err) {
                console.error('Error creating table: ' + err.message);
            }
        });
    }
});

// Function to close the database connection
const closeDb = () => {
    db.close((err) => {
        if (err) {
            console.error('Error closing database: ' + err.message);
        }
    });
};

// Export the database connection and close function
module.exports = { db, closeDb };
