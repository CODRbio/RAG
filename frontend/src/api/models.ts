import client from './client';
import type { ModelSyncRequest, ModelSyncResponse, ModelStatusResponse } from '../types';

export async function getModelStatus(): Promise<ModelStatusResponse> {
  const res = await client.get<ModelStatusResponse>('/models/status');
  return res.data;
}

export async function syncModels(payload: ModelSyncRequest = {}): Promise<ModelSyncResponse> {
  const res = await client.post<ModelSyncResponse>('/models/sync', payload);
  return res.data;
}
