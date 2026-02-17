import client from './client';
import type { Project } from '../types';

export async function listProjects(
  includeArchived = false
): Promise<Project[]> {
  const res = await client.get<Project[]>('/projects', {
    params: { include_archived: includeArchived },
  });
  return res.data;
}

export async function archiveProject(
  canvasId: string
): Promise<{ canvas_id: string; archived: boolean }> {
  const res = await client.post(`/projects/${canvasId}/archive`);
  return res.data;
}

export async function unarchiveProject(
  canvasId: string
): Promise<{ canvas_id: string; archived: boolean }> {
  const res = await client.post(`/projects/${canvasId}/unarchive`);
  return res.data;
}

export async function deleteProject(
  canvasId: string
): Promise<{ canvas_id: string; deleted: boolean }> {
  const res = await client.delete(`/projects/${canvasId}`);
  return res.data;
}
