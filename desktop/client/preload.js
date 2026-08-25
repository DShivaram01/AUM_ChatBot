const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('aumApp', {
  getServerConfig: () => ipcRenderer.invoke('get-server-config'),
  setServerConfig: (cfg) => ipcRenderer.invoke('set-server-config', cfg),
  getChats: () => ipcRenderer.invoke('get-chats'),
  setChats: (chats) => ipcRenderer.invoke('set-chats', chats),
  getProjects: () => ipcRenderer.invoke('get-projects'),
  setProjects: (projects) => ipcRenderer.invoke('set-projects', projects),
});
