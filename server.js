const express = require("express");
const app = express();
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(cors());
app.use(bodyParser.json());

// Example route
app.post("/chat", (req, res) => {
  const userMessage = req.body.message;
  // Process the message and send a response (this is a simple example)
  res.json({ reply: "This is a response to: " + userMessage });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
