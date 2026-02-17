import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Shield, Users, UserPlus, Trash2 } from 'lucide-react';
import { useUIStore, useToastStore } from '../stores';
import { listUsers, createUser } from '../api/auth';
import { Modal } from '../components/ui/Modal';
import type { UserItem } from '../types';

export function AdminPage() {
  const { t } = useTranslation();
  const { showUserModal, setShowUserModal } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);

  const [users, setUsers] = useState<UserItem[]>([]);
  const [, setIsLoading] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('user');

  const fetchUsers = async () => {
    setIsLoading(true);
    try {
      const data = await listUsers();
      setUsers(data);
    } catch {
      addToast(t('admin.fetchUsersFailed'), 'error');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleCreateUser = async () => {
    if (!newUsername.trim() || !newPassword.trim()) return;

    try {
      await createUser({
        user_id: newUsername,
        password: newPassword,
        role: newRole,
      });
      addToast(t('admin.userCreated', { name: newUsername }), 'success');
      setShowUserModal(false);
      setNewUsername('');
      setNewPassword('');
      setNewRole('user');
      fetchUsers();
    } catch {
      addToast(t('admin.createFailed'), 'error');
    }
  };

  const handleDeleteUser = (userId: string) => {
    if (userId === 'admin') {
      addToast(t('admin.cannotDeleteAdmin'), 'error');
      return;
    }
    // 删除用户 API（需要后端支持）
    addToast(t('admin.deleteInDev'), 'info');
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 p-8 animate-in fade-in duration-300">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-slate-100">{t('admin.title')}</h2>
          <p className="text-slate-400 mt-1">{t('admin.subtitle')}</p>
        </div>
        <button
          onClick={() => setShowUserModal(true)}
          className="flex items-center gap-2 bg-sky-600 text-white px-5 py-2.5 rounded-xl font-medium shadow-lg shadow-sky-500/20 hover:bg-sky-500 transition-all"
        >
          <UserPlus size={18} /> {t('admin.createUser')}
        </button>
      </div>

      <div className="bg-slate-900/60 border border-slate-700/50 rounded-2xl overflow-hidden shadow-sm">
        <table className="w-full text-sm">
          <thead className="bg-slate-800/50">
            <tr className="text-left text-slate-400 border-b border-slate-700/50">
              <th className="px-6 py-4 font-medium">{t('admin.usernameRole')}</th>
              <th className="px-6 py-4 font-medium">{t('common.status')}</th>
              <th className="px-6 py-4 font-medium">{t('admin.createdTime')}</th>
              <th className="px-6 py-4 font-medium text-right">{t('admin.management')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {users.map((user) => (
              <tr key={user.user_id} className="hover:bg-slate-800/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        user.role === 'admin'
                          ? 'bg-purple-900/30 text-purple-400'
                          : 'bg-sky-900/30 text-sky-400'
                      }`}
                    >
                      {user.role === 'admin' ? (
                        <Shield size={14} />
                      ) : (
                        <Users size={14} />
                      )}
                    </div>
                    <div>
                      <div className="font-bold text-slate-200">{user.user_id}</div>
                      <div className="text-xs text-slate-400 capitalize">
                        {user.role}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span
                    className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide ${
                      user.is_active
                        ? 'bg-emerald-900/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-500 border border-slate-700'
                    }`}
                  >
                    {user.is_active ? 'Active' : 'Inactive'}
                  </span>
                </td>
                <td className="px-6 py-4 text-slate-400 font-mono text-xs">
                  {user.created_at}
                </td>
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={() => handleDeleteUser(user.user_id)}
                    className="text-slate-500 hover:text-red-400 hover:bg-red-900/20 p-2 rounded-lg transition-colors"
                    title={t('admin.deleteUser')}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 新建用户 Modal */}
      <Modal
        open={showUserModal}
        onClose={() => setShowUserModal(false)}
        title={t('admin.createUser')}
        icon={<UserPlus size={20} />}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleCreateUser();
          }}
          className="space-y-4"
        >
          <div>
            <label className="text-xs font-medium text-slate-400 uppercase">
              {t('admin.username')}
            </label>
            <input
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              required
              type="text"
              className="w-full mt-1 bg-slate-900 border border-slate-700 text-slate-200 rounded-md p-2 text-sm focus:ring-1 focus:ring-sky-500 focus:border-sky-500 outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-400 uppercase">
              {t('admin.initialPassword')}
            </label>
            <input
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              type="password"
              className="w-full mt-1 bg-slate-900 border border-slate-700 text-slate-200 rounded-md p-2 text-sm focus:ring-1 focus:ring-sky-500 focus:border-sky-500 outline-none"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-400 uppercase">
              {t('admin.rolePermission')}
            </label>
            <select
              value={newRole}
              onChange={(e) => setNewRole(e.target.value)}
              className="w-full mt-1 bg-slate-900 border border-slate-700 text-slate-200 rounded-md p-2 text-sm"
            >
              <option value="user">{t('admin.normalUser')}</option>
              <option value="admin">{t('admin.adminUser')}</option>
            </select>
          </div>
          <div className="mt-6 flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowUserModal(false)}
              className="px-4 py-2 text-slate-400 hover:bg-slate-800 rounded-lg text-sm"
            >
              {t('common.cancel')}
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-500 text-sm"
            >
              {t('admin.createAccount')}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
