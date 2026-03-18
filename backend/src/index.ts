import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { chatRouter } from './routes/chat.js';
import { emotionRouter } from './routes/emotion.js';
import { recommendRouter } from './routes/recommend.js';
import { vodRouter } from './routes/vod.js';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use('/api/chat', chatRouter);
app.use('/api/emotion', emotionRouter);
app.use('/api/recommend', recommendRouter);
app.use('/api/vod', vodRouter);

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', mock: process.env.USE_MOCK_AI === 'true' });
});

app.listen(PORT, () => {
  console.log(`[마음이음] API 서버 http://localhost:${PORT}`);
  console.log(`[마음이음] Mock AI: ${process.env.USE_MOCK_AI === 'true' ? 'ON' : 'OFF'}`);
});
