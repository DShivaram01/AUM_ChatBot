const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

// --- Single instance enforcement -------------------------------------------------
// Prevents launching two copies of the client at once (mirrors the same concern
// your status doc raised for the backend).
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    const win = BrowserWindow.getAllWindows()[0];
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
}

// --- Server address persistence ---------------------------------------------------
// The server's host:port can change (dynamic port selection, or a different
// machine entirely). Store the last-used address in the user's app-data folder
// so they don't have to re-type it every launch.
const configPath = path.join(app.getPath('userData'), 'server-config.json');

function readServerConfig() {
  try {
    return JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  } catch {
    return { serverUrl: 'http://localhost:8000' };
  }
}

function writeServerConfig(cfg) {
  fs.writeFileSync(configPath, JSON.stringify(cfg, null, 2));
}

ipcMain.handle('get-server-config', () => readServerConfig());
ipcMain.handle('set-server-config', (_event, cfg) => {
  writeServerConfig(cfg);
  return readServerConfig();
});

// --- Chat history persistence ------------------------------------------------------
// Chats are stored as one JSON array in the user's app-data folder. Temporary
// chats are never written here at all (see renderer.js) -- this file only ever
// contains chats the user chose to save.
const chatsPath = path.join(app.getPath('userData'), 'chats.json');
const projectsPath = path.join(app.getPath('userData'), 'projects.json');

function readChats() {
  try {
    return JSON.parse(fs.readFileSync(chatsPath, 'utf-8'));
  } catch {
    return [];
  }
}

function writeChats(chats) {
  fs.writeFileSync(chatsPath, JSON.stringify(chats, null, 2));
}

ipcMain.handle('get-chats', () => readChats());
ipcMain.handle('set-chats', (_event, chats) => {
  writeChats(chats);
  return readChats();
});

function readProjects() {
  try {
    return JSON.parse(fs.readFileSync(projectsPath, 'utf-8'));
  } catch {
    return [];
  }
}

function writeProjects(projects) {
  fs.writeFileSync(projectsPath, JSON.stringify(projects, null, 2));
}

ipcMain.handle('get-projects', () => readProjects());
ipcMain.handle('set-projects', (_event, projects) => {
  writeProjects(projects);
  return readProjects();
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 720,
    minWidth: 720,
    minHeight: 560,
    backgroundColor: '#121212',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
