import { Router } from 'express';
import { getChatService } from '../services/AIServiceFactory.js';

export const chatRouter = Router();

chatRouter.post('/', async (req, res) => {
  try {
    const { message, history, context } = req.body;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }
    const service = getChatService();
    const result = await service.chat({
      message,
      history,
      context,
    });
    res.json(result);
  } catch (e) {
    console.error('[chat]', e);
    res.status(500).json({ error: 'Chat failed' });
  }
});
