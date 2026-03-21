import client from './client';
import type {
  ResourceNote,
  ResourceNoteCreate,
  ResourceNoteUpdate,
  ResourceRef,
  ResourceTag,
  ResourceTagUpsert,
  ResourceUserState,
  ResourceUserStateUpsert,
} from '../types';

export async function getResourceState(params: ResourceRef): Promise<ResourceUserState> {
  const res = await client.get<ResourceUserState>('/resources/state', { params });
  return res.data;
}

export async function patchResourceState(body: ResourceUserStateUpsert): Promise<ResourceUserState> {
  const res = await client.patch<ResourceUserState>('/resources/state', body);
  return res.data;
}

export async function listResourceTags(params: ResourceRef): Promise<{ items: ResourceTag[] }> {
  const res = await client.get<{ items: ResourceTag[] }>('/resources/tags', { params });
  return res.data;
}

export async function createResourceTag(body: ResourceTagUpsert): Promise<ResourceTag> {
  const res = await client.post<ResourceTag>('/resources/tags', body);
  return res.data;
}

export async function deleteResourceTag(params: ResourceTagUpsert): Promise<{ deleted: boolean }> {
  const res = await client.delete<{ deleted: boolean }>('/resources/tags', { params });
  return res.data;
}

export async function listResourceNotes(params: ResourceRef): Promise<{ items: ResourceNote[] }> {
  const res = await client.get<{ items: ResourceNote[] }>('/resources/notes', { params });
  return res.data;
}

export async function createResourceNote(body: ResourceNoteCreate): Promise<ResourceNote> {
  const res = await client.post<ResourceNote>('/resources/notes', body);
  return res.data;
}

export async function updateResourceNote(noteId: number, body: ResourceNoteUpdate): Promise<ResourceNote> {
  const res = await client.patch<ResourceNote>(`/resources/notes/${noteId}`, body);
  return res.data;
}

export async function deleteResourceNote(noteId: number): Promise<{ deleted: boolean }> {
  const res = await client.delete<{ deleted: boolean }>(`/resources/notes/${noteId}`);
  return res.data;
}
