"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.KernelBootstrapper = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs/promises"));
const https = __importStar(require("https"));
const tar = __importStar(require("tar"));
const child_process_1 = require("child_process");
const util_1 = require("util");
const semver = __importStar(require("semver"));
const execAsync = (0, util_1.promisify)(child_process_1.exec);
class KernelBootstrapper {
    constructor(context) {
        this.context = context;
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const repoSettings = config.get('repository', {
            owner: 'your-username',
            repo: 'LLMKernel',
            branch: 'main'
        });
        this.repoConfig = {
            owner: repoSettings.owner,
            repo: repoSettings.repo,
            branch: repoSettings.branch,
            kernelPath: 'kernel/',
            releasesEndpoint: `https://api.github.com/repos/${repoSettings.owner}/${repoSettings.repo}/releases`
        };
        this.kernelDir = path.join(context.globalStorageUri.fsPath, 'llm-kernel');
        this.outputChannel = vscode.window.createOutputChannel('LLM Kernel Bootstrap');
    }
    isKernelInstalled() {
        return __awaiter(this, void 0, void 0, function* () {
            // Check if installed in global storage (remote/release install)
            try {
                yield fs.access(this.kernelDir);
                const kernelFile = path.join(this.kernelDir, 'llm_kernel.py');
                yield fs.access(kernelFile);
                return true;
            }
            catch (_a) {
                // Not in global storage
            }
            // Check if kernel is registered with Jupyter (covers local editable install)
            try {
                const { stdout } = yield execAsync('jupyter kernelspec list');
                if (stdout.toLowerCase().includes('llm_kernel') || stdout.toLowerCase().includes('llm-kernel')) {
                    return true;
                }
            }
            catch (_b) {
                // jupyter not found or failed
            }
            return false;
        });
    }
    bootstrapFromRepository() {
        return __awaiter(this, void 0, void 0, function* () {
            yield vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Bootstrapping LLM Kernel from repository...",
                cancellable: false
            }, (progress) => __awaiter(this, void 0, void 0, function* () {
                try {
                    // Ensure kernel directory exists
                    yield fs.mkdir(this.kernelDir, { recursive: true });
                    progress.report({ increment: 10, message: "Checking for latest version..." });
                    const latestRelease = yield this.getLatestRelease();
                    progress.report({ increment: 30, message: "Downloading kernel files..." });
                    yield this.downloadKernelFromRepo(latestRelease);
                    progress.report({ increment: 60, message: "Installing dependencies..." });
                    yield this.installDependencies();
                    progress.report({ increment: 80, message: "Registering kernel..." });
                    yield this.registerKernel();
                    progress.report({ increment: 100, message: "Bootstrap complete!" });
                    vscode.window.showInformationMessage('🚀 LLM Kernel installed successfully! Use Ctrl+Shift+L in any notebook to activate LLM features.');
                }
                catch (error) {
                    this.outputChannel.appendLine(`Bootstrap failed: ${error}`);
                    this.outputChannel.show();
                    throw error;
                }
            }));
        });
    }
    /**
     * Check if we're in local development mode.
     * True when the workspace contains llm_kernel/ and pyproject.toml,
     * indicating the monorepo is open and we can install from local source.
     */
    isLocalDevelopment() {
        return __awaiter(this, void 0, void 0, function* () {
            const repoRoot = yield this.getLocalRepoRoot();
            return repoRoot !== null;
        });
    }
    /**
     * Get the root path of the monorepo workspace containing llm_kernel/.
     * Returns null if not in local development mode.
     */
    getLocalRepoRoot() {
        return __awaiter(this, void 0, void 0, function* () {
            const workspaceFolders = vscode.workspace.workspaceFolders;
            if (!workspaceFolders) {
                return null;
            }
            for (const folder of workspaceFolders) {
                const localKernelPath = path.join(folder.uri.fsPath, 'llm_kernel');
                const pyprojectPath = path.join(folder.uri.fsPath, 'pyproject.toml');
                try {
                    yield fs.access(localKernelPath);
                    yield fs.access(pyprojectPath);
                    return folder.uri.fsPath;
                }
                catch (_a) {
                    continue;
                }
            }
            return null;
        });
    }
    /**
     * Bootstrap the kernel from local source code.
     * Uses pip install -e . for editable install, then registers the kernel spec.
     */
    bootstrapFromLocal() {
        return __awaiter(this, void 0, void 0, function* () {
            const repoRoot = yield this.getLocalRepoRoot();
            if (!repoRoot) {
                throw new Error('Local repository root not found');
            }
            yield vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Installing LLM Kernel from local source...",
                cancellable: false
            }, (progress) => __awaiter(this, void 0, void 0, function* () {
                try {
                    const pythonPath = yield this.findPythonPath();
                    progress.report({ increment: 20, message: "Installing in editable mode..." });
                    this.outputChannel.appendLine(`Installing from local source: ${repoRoot}`);
                    // Run pip install -e . from the repo root
                    const { stdout, stderr } = yield execAsync(`"${pythonPath}" -m pip install -e .`, { cwd: repoRoot });
                    this.outputChannel.appendLine(stdout);
                    if (stderr) {
                        this.outputChannel.appendLine(`Warnings: ${stderr}`);
                    }
                    progress.report({ increment: 60, message: "Registering kernel with Jupyter..." });
                    // Use the kernel's own install module to register the kernel spec
                    const { stdout: installOut, stderr: installErr } = yield execAsync(`"${pythonPath}" -m llm_kernel.install install --user`, { cwd: repoRoot });
                    this.outputChannel.appendLine(installOut);
                    if (installErr) {
                        this.outputChannel.appendLine(`Warnings: ${installErr}`);
                    }
                    progress.report({ increment: 100, message: "Local installation complete!" });
                    vscode.window.showInformationMessage('LLM Kernel installed from local source (editable mode). Changes to llm_kernel/ take effect immediately.');
                }
                catch (error) {
                    this.outputChannel.appendLine(`Local bootstrap failed: ${error}`);
                    this.outputChannel.show();
                    throw error;
                }
            }));
        });
    }
    getLatestRelease() {
        return __awaiter(this, void 0, void 0, function* () {
            const updateChannel = vscode.workspace.getConfiguration('llm-kernel').get('updateChannel', 'stable');
            try {
                const response = yield this.fetchJson(this.repoConfig.releasesEndpoint);
                const releases = response;
                if (releases.length === 0) {
                    throw new Error('No releases found in repository');
                }
                // Filter releases based on update channel
                const filteredReleases = releases.filter(release => {
                    if (updateChannel === 'stable' && !release.tag_name.includes('beta') && !release.tag_name.includes('dev')) {
                        return true;
                    }
                    else if (updateChannel === 'beta' && release.tag_name.includes('beta')) {
                        return true;
                    }
                    else if (updateChannel === 'dev') {
                        return true;
                    }
                    return false;
                });
                if (filteredReleases.length === 0) {
                    return releases[0]; // Fallback to latest release
                }
                return filteredReleases[0];
            }
            catch (error) {
                this.outputChannel.appendLine(`Failed to get releases: ${error}`);
                throw new Error(`Failed to fetch releases from GitHub: ${error}`);
            }
        });
    }
    downloadKernelFromRepo(release) {
        return __awaiter(this, void 0, void 0, function* () {
            this.outputChannel.appendLine(`Downloading kernel version ${release.tag_name}...`);
            // Look for pre-built kernel bundle in release assets
            const kernelAsset = release.assets.find(asset => asset.name.includes('llm-kernel') && (asset.name.endsWith('.tar.gz') || asset.name.endsWith('.zip')));
            if (kernelAsset) {
                // Download pre-built bundle
                yield this.downloadAndExtractAsset(kernelAsset.browser_download_url, kernelAsset.name);
            }
            else {
                // Download specific files from repository
                yield this.downloadKernelFiles();
            }
            // Save version information
            const versionFile = path.join(this.kernelDir, 'VERSION');
            yield fs.writeFile(versionFile, release.tag_name);
        });
    }
    downloadAndExtractAsset(url, filename) {
        return __awaiter(this, void 0, void 0, function* () {
            const downloadPath = path.join(this.kernelDir, filename);
            yield this.downloadFile(url, downloadPath);
            if (filename.endsWith('.tar.gz')) {
                yield tar.extract({
                    file: downloadPath,
                    cwd: this.kernelDir
                });
            }
            else if (filename.endsWith('.zip')) {
                // Use VS Code's built-in unzip if available, or fallback to node-unzip
                yield execAsync(`unzip -o "${downloadPath}" -d "${this.kernelDir}"`);
            }
            // Clean up downloaded archive
            yield fs.unlink(downloadPath);
        });
    }
    downloadKernelFiles() {
        return __awaiter(this, void 0, void 0, function* () {
            const kernelFiles = [
                'llm_kernel.py',
                'requirements.txt',
                'kernel.json',
                'context_manager.py',
                'llm_interface.py',
                'kernel_proxy.py',
                '__init__.py'
            ];
            for (const file of kernelFiles) {
                try {
                    yield this.downloadFileFromRepo(file);
                }
                catch (error) {
                    this.outputChannel.appendLine(`Warning: Could not download ${file}: ${error}`);
                }
            }
        });
    }
    downloadFileFromRepo(filePath) {
        return __awaiter(this, void 0, void 0, function* () {
            const url = `https://api.github.com/repos/${this.repoConfig.owner}/${this.repoConfig.repo}/contents/${this.repoConfig.kernelPath}${filePath}?ref=${this.repoConfig.branch}`;
            const response = yield this.fetchJson(url);
            if (response.content) {
                // GitHub API returns base64-encoded content
                const content = Buffer.from(response.content, 'base64');
                const targetPath = path.join(this.kernelDir, filePath);
                // Ensure directory exists
                const dir = path.dirname(targetPath);
                yield fs.mkdir(dir, { recursive: true });
                yield fs.writeFile(targetPath, content);
                // Make Python files executable
                if (filePath.endsWith('.py')) {
                    yield fs.chmod(targetPath, 0o755);
                }
            }
        });
    }
    installDependencies() {
        return __awaiter(this, void 0, void 0, function* () {
            this.outputChannel.appendLine('Installing Python dependencies...');
            const pythonPath = yield this.findPythonPath();
            const requirementsPath = path.join(this.kernelDir, 'requirements.txt');
            try {
                // Check if requirements.txt exists
                yield fs.access(requirementsPath);
                // Install dependencies
                const { stdout, stderr } = yield execAsync(`"${pythonPath}" -m pip install -r "${requirementsPath}" --user`, { cwd: this.kernelDir });
                this.outputChannel.appendLine(stdout);
                if (stderr) {
                    this.outputChannel.appendLine(`Warnings: ${stderr}`);
                }
            }
            catch (error) {
                this.outputChannel.appendLine(`Failed to install dependencies: ${error}`);
                throw new Error(`Failed to install Python dependencies: ${error}`);
            }
        });
    }
    registerKernel() {
        return __awaiter(this, void 0, void 0, function* () {
            this.outputChannel.appendLine('Registering kernel with Jupyter...');
            try {
                // Check if Jupyter is installed
                const { stdout: jupyterCheck } = yield execAsync('jupyter --version');
                this.outputChannel.appendLine(`Jupyter version: ${jupyterCheck}`);
                // Update kernel.json with correct Python path
                const kernelSpecPath = path.join(this.kernelDir, 'kernel.json');
                const kernelSpec = JSON.parse(yield fs.readFile(kernelSpecPath, 'utf-8'));
                const pythonPath = yield this.findPythonPath();
                kernelSpec.argv[0] = pythonPath;
                yield fs.writeFile(kernelSpecPath, JSON.stringify(kernelSpec, null, 2));
                // Install kernel spec
                const { stdout, stderr } = yield execAsync(`jupyter kernelspec install "${this.kernelDir}" --name llm-kernel --user`);
                this.outputChannel.appendLine(stdout);
                if (stderr) {
                    this.outputChannel.appendLine(`Warnings: ${stderr}`);
                }
                this.outputChannel.appendLine('Kernel registered successfully!');
            }
            catch (error) {
                this.outputChannel.appendLine(`Failed to register kernel: ${error}`);
                throw new Error(`Failed to register kernel with Jupyter: ${error}`);
            }
        });
    }
    findPythonPath() {
        return __awaiter(this, void 0, void 0, function* () {
            // Try multiple methods to find Python
            const pythonCommands = ['python3', 'python', 'py'];
            const whichCommand = process.platform === 'win32' ? 'where' : 'which';
            for (const cmd of pythonCommands) {
                try {
                    const { stdout } = yield execAsync(`${cmd} --version`);
                    if (stdout.includes('Python')) {
                        const { stdout: pythonPath } = yield execAsync(`${whichCommand} ${cmd}`);
                        // 'where' on Windows may return multiple lines; take the first
                        return pythonPath.trim().split(/\r?\n/)[0].trim();
                    }
                }
                catch (_a) {
                    continue;
                }
            }
            // Try VS Code's Python extension
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            if (pythonExtension) {
                try {
                    const pythonPath = yield pythonExtension.exports.settings.getExecutionDetails().execCommand;
                    if (pythonPath) {
                        return pythonPath[0];
                    }
                }
                catch (_b) {
                    // Python extension API may have changed
                }
            }
            throw new Error('Python interpreter not found. Please ensure Python is installed and accessible.');
        });
    }
    downloadFile(url, targetPath) {
        return __awaiter(this, void 0, void 0, function* () {
            return new Promise((resolve, reject) => {
                const file = require('fs').createWriteStream(targetPath);
                https.get(url, { headers: { 'User-Agent': 'VSCode-LLM-Kernel' } }, (response) => {
                    if (response.statusCode === 302 || response.statusCode === 301) {
                        // Follow redirect
                        this.downloadFile(response.headers.location, targetPath)
                            .then(resolve)
                            .catch(reject);
                        return;
                    }
                    if (response.statusCode !== 200) {
                        reject(new Error(`Failed to download: ${response.statusCode}`));
                        return;
                    }
                    response.pipe(file);
                    file.on('finish', () => {
                        file.close();
                        resolve();
                    });
                }).on('error', reject);
            });
        });
    }
    fetchJson(url) {
        return __awaiter(this, void 0, void 0, function* () {
            return new Promise((resolve, reject) => {
                https.get(url, { headers: { 'User-Agent': 'VSCode-LLM-Kernel' } }, (response) => {
                    let data = '';
                    if (response.statusCode !== 200) {
                        reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`));
                        return;
                    }
                    response.on('data', chunk => data += chunk);
                    response.on('end', () => {
                        try {
                            resolve(JSON.parse(data));
                        }
                        catch (error) {
                            reject(error);
                        }
                    });
                }).on('error', reject);
            });
        });
    }
    getCurrentVersion() {
        return __awaiter(this, void 0, void 0, function* () {
            const versionFile = path.join(this.kernelDir, 'VERSION');
            try {
                const version = yield fs.readFile(versionFile, 'utf-8');
                return version.trim();
            }
            catch (_a) {
                return '0.0.0';
            }
        });
    }
    checkForUpdates() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const currentVersion = yield this.getCurrentVersion();
                const latestRelease = yield this.getLatestRelease();
                const latestVersion = latestRelease.tag_name.replace(/^v/, '');
                return semver.gt(latestVersion, currentVersion);
            }
            catch (error) {
                this.outputChannel.appendLine(`Failed to check for updates: ${error}`);
                return false;
            }
        });
    }
    performUpdate() {
        return __awaiter(this, void 0, void 0, function* () {
            yield vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Updating LLM Kernel...",
                cancellable: false
            }, (progress) => __awaiter(this, void 0, void 0, function* () {
                try {
                    // Backup current installation
                    progress.report({ increment: 20, message: "Backing up current installation..." });
                    yield this.backupCurrentInstallation();
                    // Download new version
                    progress.report({ increment: 50, message: "Downloading latest version..." });
                    const latestRelease = yield this.getLatestRelease();
                    yield this.downloadKernelFromRepo(latestRelease);
                    // Update dependencies if needed
                    progress.report({ increment: 80, message: "Updating dependencies..." });
                    yield this.installDependencies();
                    // Re-register kernel
                    progress.report({ increment: 90, message: "Updating kernel registration..." });
                    yield this.registerKernel();
                    progress.report({ increment: 100, message: "Update complete!" });
                    vscode.window.showInformationMessage(`✅ LLM Kernel updated successfully to ${latestRelease.tag_name}!`);
                }
                catch (error) {
                    this.outputChannel.appendLine(`Update failed: ${error}`);
                    this.outputChannel.show();
                    // Restore backup on failure
                    yield this.restoreBackup();
                    throw error;
                }
            }));
        });
    }
    backupCurrentInstallation() {
        return __awaiter(this, void 0, void 0, function* () {
            const backupDir = path.join(this.context.globalStorageUri.fsPath, 'llm-kernel-backup');
            try {
                // Remove old backup if exists
                yield fs.rm(backupDir, { recursive: true, force: true });
                // Copy current installation to backup
                yield this.copyDirectory(this.kernelDir, backupDir);
            }
            catch (error) {
                this.outputChannel.appendLine(`Backup warning: ${error}`);
            }
        });
    }
    restoreBackup() {
        return __awaiter(this, void 0, void 0, function* () {
            const backupDir = path.join(this.context.globalStorageUri.fsPath, 'llm-kernel-backup');
            try {
                yield fs.rm(this.kernelDir, { recursive: true, force: true });
                yield this.copyDirectory(backupDir, this.kernelDir);
                this.outputChannel.appendLine('Restored from backup successfully');
            }
            catch (error) {
                this.outputChannel.appendLine(`Failed to restore backup: ${error}`);
            }
        });
    }
    copyDirectory(source, destination) {
        return __awaiter(this, void 0, void 0, function* () {
            yield fs.mkdir(destination, { recursive: true });
            const entries = yield fs.readdir(source, { withFileTypes: true });
            for (const entry of entries) {
                const sourcePath = path.join(source, entry.name);
                const destPath = path.join(destination, entry.name);
                if (entry.isDirectory()) {
                    yield this.copyDirectory(sourcePath, destPath);
                }
                else {
                    yield fs.copyFile(sourcePath, destPath);
                }
            }
        });
    }
}
exports.KernelBootstrapper = KernelBootstrapper;
