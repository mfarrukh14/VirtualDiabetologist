// database.js
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Initialize the database
const db = new sqlite3.Database(path.join(__dirname, 'prod_database.db'), (err) => {
    if (err) {
        console.error('Error opening database ' + err.message);
    } else {
        // Create the tables if they don't exist
        db.run(`CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            createdOn TEXT NOT NULL,
            expireOn TEXT NOT NULL
        );`, (err) => {
            if (err) {
                console.error('Error creating table: ' + err.message);
            }
        });

        db.run(`CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            createdAt TEXT NOT NULL
        );`, (err) => {
            if (err) {
                console.error('Error creating chat history table: ' + err.message);
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
