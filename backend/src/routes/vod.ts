import { Router } from 'express';
import { getVODAnalysisService } from '../services/AIServiceFactory.js';
import { VOD_CATALOG } from '../data/vodCatalog.js';

export const vodRouter = Router();

vodRouter.get('/catalog', (_req, res) => {
  res.json({ items: VOD_CATALOG });
});

vodRouter.post('/analysis', async (req, res) => {
  try {
    const { vodId, vodTitle, genre } = req.body;
    if (!vodId || !vodTitle) {
      return res.status(400).json({ error: 'vodId and vodTitle are required' });
    }
    const service = getVODAnalysisService();
    const result = await service.analyze({
      vodId,
      vodTitle,
      genre: genre || '',
    });
    res.json(result);
  } catch (e) {
    console.error('[vod-analysis]', e);
    res.status(500).json({ error: 'VOD analysis failed' });
  }
});
