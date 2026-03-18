import { Router } from 'express';
import { getRecommendService } from '../services/AIServiceFactory.js';

export const recommendRouter = Router();

recommendRouter.post('/', async (req, res) => {
  try {
    const { emotion, watchHistory, lastVodId, limit } = req.body;
    const service = getRecommendService();
    const result = await service.recommend({
      emotion,
      watchHistory: watchHistory || [],
      lastVodId,
      limit: limit || 6,
    });
    res.json(result);
  } catch (e) {
    console.error('[recommend]', e);
    res.status(500).json({ error: 'Recommend failed' });
  }
});
