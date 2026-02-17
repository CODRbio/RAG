import client from './client';
import type { LoginRequest, LoginResponse, UserItem } from '../types';

export async function login(data: LoginRequest): Promise<LoginResponse> {
  const res = await client.post<LoginResponse>('/auth/login', data);
  return res.data;
}

export async function listUsers(): Promise<UserItem[]> {
  const res = await client.get<UserItem[]>('/admin/users');
  return res.data;
}

export async function createUser(data: {
  user_id: string;
  password: string;
  role: string;
}): Promise<{ user_id: string; role: string }> {
  const res = await client.post('/admin/users', data);
  return res.data;
}
