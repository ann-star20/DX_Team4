import { Router } from 'express';
import { getEmotionService } from '../services/AIServiceFactory.js';

export const emotionRouter = Router();

emotionRouter.post('/', async (req, res) => {
  try {
    const { text, voiceTone } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'text is required' });
    }
    const service = getEmotionService();
    const result = await service.analyze({ text, voiceTone });
    res.json(result);
  } catch (e) {
    console.error('[emotion]', e);
    res.status(500).json({ error: 'Emotion analysis failed' });
  }
});
