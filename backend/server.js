import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const FLASK_URL = "http://localhost:5000";

// ---- Predict Emotion ----
app.post("/api/predict", async (req, res) => {
  try {
    const response = await fetch(`${FLASK_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---- Federated Update ----
app.post("/api/federated-update", async (req, res) => {
  try {
    const response = await fetch(`${FLASK_URL}/federated-update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/", (req, res) => {
  res.send("Node backend running!");
});

const PORT = 8080;
app.listen(PORT, () => console.log(`✅ Node server running on port ${PORT}`));
