// server.js
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(express.json({ limit: "1mb" }));

/** Allow your front-ends here */
app.use(cors({
  origin: [
    "http://localhost:3000",
    "https://second-brain-3j3g.onrender.com",      // same host (Render shows this as allowed anyway)
    // add your static site domain here if you host index.html elsewhere
    "*"
  ],
  methods: ["GET","POST","OPTIONS"],
}));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small";
const CHAT_MODEL  = process.env.OPENAI_CHAT_MODEL  || "gpt-4o-mini";

function cosine(a, b) {
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) || 1);
}

app.get("/health", (_, res) => res.json({ ok: true }));

app.post("/api/ask", async (req, res) => {
  try {
    const { question, notes } = req.body || {};
    if (!question || !Array.isArray(notes)) {
      return res.status(400).json({ ok:false, error:"question and notes[] required" });
    }

    // 1) embed the question
    const qEmb = await openai.embeddings.create({ model: EMBED_MODEL, input: question });
    const qVec = qEmb.data[0].embedding;

    // 2) embed each note text and score
    const texts = notes.map(n => `${n.title || ""}\n\n${n.content || ""}`.trim());
    const emb = await openai.embeddings.create({ model: EMBED_MODEL, input: texts });
    const scored = texts.map((t,i) => ({
      title: notes[i].title || "",
      text: t,
      score: cosine(qVec, emb.data[i].embedding)
    }))
    .sort((a,b)=>b.score - a.score)
    .slice(0, 6);

    const context = scored
      .map((n,i)=>`[${i+1}] ${n.title || "(no title)"}: ${n.text.slice(0, 1200)}`)
      .join("\n\n");

    // 3) chat with context
    const sys = "You are the user's private second brain. Answer ONLY from the context. If unsure, say you don't know. Cite like [1], [2].";
    const chat = await openai.chat.completions.create({
      model: CHAT_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: sys },
        { role: "user", content: `Question: ${question}\n\nContext:\n${context}` }
      ]
    });

    const answer = chat.choices[0]?.message?.content || "";
    const sources = scored.map(s => ({ title: s.title }));
    res.json({ ok:true, answer, sources });
  } catch (e) {
    res.status(500).json({ ok:false, error: e?.message || "ask failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("API up on :" + PORT));
