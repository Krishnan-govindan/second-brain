// server.js
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
app.use(express.json({ limit: "1mb" }));

// Resolve __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Allow your frontend origins
app.use(cors({
  origin: "*",  // for testing; later restrict to your domain
  methods: ["GET", "POST", "OPTIONS"],
}));

// Serve static files from /public
app.use(express.static(path.join(__dirname, "public")));

// Root route -> index.html
app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// OpenAI setup
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small";
const CHAT_MODEL  = process.env.OPENAI_CHAT_MODEL  || "gpt-4o-mini";

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// Health check
app.get("/health", (_req, res) => res.json({ ok: true }));

// Ask route
app.post("/api/ask", async (req, res) => {
  try {
    const { question, notes } = req.body || {};
    if (!question || !Array.isArray(notes)) {
      return res.status(400).json({ ok: false, error: "question and notes[] required" });
    }

    // Embed question
    const qEmb = await openai.embeddings.create({ model: EMBED_MODEL, input: question });
    const qVec = qEmb.data[0].embedding;

    // Embed notes
    const texts = notes.map(n => `${n.title || ""}\n\n${n.content || ""}`.trim());
    const emb = await openai.embeddings.create({ model: EMBED_MODEL, input: texts });
    const scored = texts.map((t, i) => ({
      title: notes[i].title || "",
      text: t,
      score: cosine(qVec, emb.data[i].embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

    const context = scored
      .map((n, i) => `[${i+1}] ${n.title || "(no title)"}: ${n.text.slice(0, 1200)}`)
      .join("\n\n");

    // Ask OpenAI
    const chat = await openai.chat.completions.create({
      model: CHAT_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: "You are the user's private second brain. Only answer from context." },
        { role: "user", content: `Question: ${question}\n\nContext:\n${context}` }
      ]
    });

    const answer = chat.choices[0]?.message?.content || "";
    const sources = scored.map(s => ({ title: s.title }));

    res.json({ ok: true, answer, sources });
  } catch (e) {
    res.status(500).json({ ok: false, error: e?.message || "ask failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("API + Static server running on :" + PORT));
