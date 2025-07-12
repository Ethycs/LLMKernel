import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as https from 'https';
import * as tar from 'tar';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as semver from 'semver';

const execAsync = promisify(exec);

interface KernelRepoConfig {
    owner: string;
    repo: string;
    branch: string;
    kernelPath: string;
    releasesEndpoint: string;
}

interface GitHubRelease {
    tag_name: string;
    tarball_url: string;
    zipball_url: string;
    assets: Array<{
        name: string;
        browser_download_url: string;
    }>;
    published_at: string;
}

export class KernelBootstrapper {
    private readonly repoConfig: KernelRepoConfig;
    private readonly kernelDir: string;
    private outputChannel: vscode.OutputChannel;

    constructor(private context: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('llm-kernel');
        const repoSettings = config.get<any>('repository', {
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

    async isKernelInstalled(): Promise<boolean> {
        try {
            await fs.access(this.kernelDir);
            const kernelFile = path.join(this.kernelDir, 'llm_kernel.py');
            await fs.access(kernelFile);
            return true;
        } catch {
            return false;
        }
    }

    async bootstrapFromRepository(): Promise<void> {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Bootstrapping LLM Kernel from repository...",
            cancellable: false
        }, async (progress) => {
            try {
                // Ensure kernel directory exists
                await fs.mkdir(this.kernelDir, { recursive: true });

                progress.report({ increment: 10, message: "Checking for latest version..." });
                const latestRelease = await this.getLatestRelease();

                progress.report({ increment: 30, message: "Downloading kernel files..." });
                await this.downloadKernelFromRepo(latestRelease);

                progress.report({ increment: 60, message: "Installing dependencies..." });
                await this.installDependencies();

                progress.report({ increment: 80, message: "Registering kernel..." });
                await this.registerKernel();

                progress.report({ increment: 100, message: "Bootstrap complete!" });

                vscode.window.showInformationMessage(
                    '🚀 LLM Kernel installed successfully! Use Ctrl+Shift+L in any notebook to activate LLM features.'
                );
            } catch (error) {
                this.outputChannel.appendLine(`Bootstrap failed: ${error}`);
                this.outputChannel.show();
                throw error;
            }
        });
    }

    private async getLatestRelease(): Promise<GitHubRelease> {
        const updateChannel = vscode.workspace.getConfiguration('llm-kernel').get<string>('updateChannel', 'stable');
        
        try {
            const response = await this.fetchJson(this.repoConfig.releasesEndpoint);
            const releases = response as GitHubRelease[];

            if (releases.length === 0) {
                throw new Error('No releases found in repository');
            }

            // Filter releases based on update channel
            const filteredReleases = releases.filter(release => {
                if (updateChannel === 'stable' && !release.tag_name.includes('beta') && !release.tag_name.includes('dev')) {
                    return true;
                } else if (updateChannel === 'beta' && release.tag_name.includes('beta')) {
                    return true;
                } else if (updateChannel === 'dev') {
                    return true;
                }
                return false;
            });

            if (filteredReleases.length === 0) {
                return releases[0]; // Fallback to latest release
            }

            return filteredReleases[0];
        } catch (error) {
            this.outputChannel.appendLine(`Failed to get releases: ${error}`);
            throw new Error(`Failed to fetch releases from GitHub: ${error}`);
        }
    }

    private async downloadKernelFromRepo(release: GitHubRelease): Promise<void> {
        this.outputChannel.appendLine(`Downloading kernel version ${release.tag_name}...`);

        // Look for pre-built kernel bundle in release assets
        const kernelAsset = release.assets.find(asset => 
            asset.name.includes('llm-kernel') && (asset.name.endsWith('.tar.gz') || asset.name.endsWith('.zip'))
        );

        if (kernelAsset) {
            // Download pre-built bundle
            await this.downloadAndExtractAsset(kernelAsset.browser_download_url, kernelAsset.name);
        } else {
            // Download specific files from repository
            await this.downloadKernelFiles();
        }

        // Save version information
        const versionFile = path.join(this.kernelDir, 'VERSION');
        await fs.writeFile(versionFile, release.tag_name);
    }

    private async downloadAndExtractAsset(url: string, filename: string): Promise<void> {
        const downloadPath = path.join(this.kernelDir, filename);
        
        await this.downloadFile(url, downloadPath);

        if (filename.endsWith('.tar.gz')) {
            await tar.extract({
                file: downloadPath,
                cwd: this.kernelDir
            });
        } else if (filename.endsWith('.zip')) {
            // Use VS Code's built-in unzip if available, or fallback to node-unzip
            await execAsync(`unzip -o "${downloadPath}" -d "${this.kernelDir}"`);
        }

        // Clean up downloaded archive
        await fs.unlink(downloadPath);
    }

    private async downloadKernelFiles(): Promise<void> {
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
                await this.downloadFileFromRepo(file);
            } catch (error) {
                this.outputChannel.appendLine(`Warning: Could not download ${file}: ${error}`);
            }
        }
    }

    private async downloadFileFromRepo(filePath: string): Promise<void> {
        const url = `https://api.github.com/repos/${this.repoConfig.owner}/${this.repoConfig.repo}/contents/${this.repoConfig.kernelPath}${filePath}?ref=${this.repoConfig.branch}`;
        
        const response = await this.fetchJson(url);
        
        if (response.content) {
            // GitHub API returns base64-encoded content
            const content = Buffer.from(response.content, 'base64');
            const targetPath = path.join(this.kernelDir, filePath);
            
            // Ensure directory exists
            const dir = path.dirname(targetPath);
            await fs.mkdir(dir, { recursive: true });
            
            await fs.writeFile(targetPath, content);
            
            // Make Python files executable
            if (filePath.endsWith('.py')) {
                await fs.chmod(targetPath, 0o755);
            }
        }
    }

    private async installDependencies(): Promise<void> {
        this.outputChannel.appendLine('Installing Python dependencies...');

        const pythonPath = await this.findPythonPath();
        const requirementsPath = path.join(this.kernelDir, 'requirements.txt');

        try {
            // Check if requirements.txt exists
            await fs.access(requirementsPath);

            // Install dependencies
            const { stdout, stderr } = await execAsync(
                `"${pythonPath}" -m pip install -r "${requirementsPath}" --user`,
                { cwd: this.kernelDir }
            );

            this.outputChannel.appendLine(stdout);
            if (stderr) {
                this.outputChannel.appendLine(`Warnings: ${stderr}`);
            }
        } catch (error) {
            this.outputChannel.appendLine(`Failed to install dependencies: ${error}`);
            throw new Error(`Failed to install Python dependencies: ${error}`);
        }
    }

    private async registerKernel(): Promise<void> {
        this.outputChannel.appendLine('Registering kernel with Jupyter...');

        try {
            // Check if Jupyter is installed
            const { stdout: jupyterCheck } = await execAsync('jupyter --version');
            this.outputChannel.appendLine(`Jupyter version: ${jupyterCheck}`);

            // Update kernel.json with correct Python path
            const kernelSpecPath = path.join(this.kernelDir, 'kernel.json');
            const kernelSpec = JSON.parse(await fs.readFile(kernelSpecPath, 'utf-8'));
            
            const pythonPath = await this.findPythonPath();
            kernelSpec.argv[0] = pythonPath;
            
            await fs.writeFile(kernelSpecPath, JSON.stringify(kernelSpec, null, 2));

            // Install kernel spec
            const { stdout, stderr } = await execAsync(
                `jupyter kernelspec install "${this.kernelDir}" --name llm-kernel --user`
            );

            this.outputChannel.appendLine(stdout);
            if (stderr) {
                this.outputChannel.appendLine(`Warnings: ${stderr}`);
            }

            this.outputChannel.appendLine('Kernel registered successfully!');
        } catch (error) {
            this.outputChannel.appendLine(`Failed to register kernel: ${error}`);
            throw new Error(`Failed to register kernel with Jupyter: ${error}`);
        }
    }

    private async findPythonPath(): Promise<string> {
        // Try multiple methods to find Python
        const pythonCommands = ['python3', 'python', 'py'];
        
        for (const cmd of pythonCommands) {
            try {
                const { stdout } = await execAsync(`${cmd} --version`);
                if (stdout.includes('Python')) {
                    const { stdout: pythonPath } = await execAsync(`which ${cmd}` || `where ${cmd}`);
                    return pythonPath.trim();
                }
            } catch {
                continue;
            }
        }

        // Try VS Code's Python extension
        const pythonExtension = vscode.extensions.getExtension('ms-python.python');
        if (pythonExtension) {
            const pythonPath = await pythonExtension.exports.settings.getExecutionDetails().execCommand;
            if (pythonPath) {
                return pythonPath[0];
            }
        }

        throw new Error('Python interpreter not found. Please ensure Python is installed and accessible.');
    }

    private async downloadFile(url: string, targetPath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const file = require('fs').createWriteStream(targetPath);
            
            https.get(url, { headers: { 'User-Agent': 'VSCode-LLM-Kernel' } }, (response) => {
                if (response.statusCode === 302 || response.statusCode === 301) {
                    // Follow redirect
                    this.downloadFile(response.headers.location!, targetPath)
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
    }

    private async fetchJson(url: string): Promise<any> {
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
                    } catch (error) {
                        reject(error);
                    }
                });
            }).on('error', reject);
        });
    }

    async getCurrentVersion(): Promise<string> {
        const versionFile = path.join(this.kernelDir, 'VERSION');
        try {
            const version = await fs.readFile(versionFile, 'utf-8');
            return version.trim();
        } catch {
            return '0.0.0';
        }
    }

    async checkForUpdates(): Promise<boolean> {
        try {
            const currentVersion = await this.getCurrentVersion();
            const latestRelease = await this.getLatestRelease();
            const latestVersion = latestRelease.tag_name.replace(/^v/, '');

            return semver.gt(latestVersion, currentVersion);
        } catch (error) {
            this.outputChannel.appendLine(`Failed to check for updates: ${error}`);
            return false;
        }
    }

    async performUpdate(): Promise<void> {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Updating LLM Kernel...",
            cancellable: false
        }, async (progress) => {
            try {
                // Backup current installation
                progress.report({ increment: 20, message: "Backing up current installation..." });
                await this.backupCurrentInstallation();

                // Download new version
                progress.report({ increment: 50, message: "Downloading latest version..." });
                const latestRelease = await this.getLatestRelease();
                await this.downloadKernelFromRepo(latestRelease);

                // Update dependencies if needed
                progress.report({ increment: 80, message: "Updating dependencies..." });
                await this.installDependencies();

                // Re-register kernel
                progress.report({ increment: 90, message: "Updating kernel registration..." });
                await this.registerKernel();

                progress.report({ increment: 100, message: "Update complete!" });

                vscode.window.showInformationMessage(
                    `✅ LLM Kernel updated successfully to ${latestRelease.tag_name}!`
                );
            } catch (error) {
                this.outputChannel.appendLine(`Update failed: ${error}`);
                this.outputChannel.show();
                
                // Restore backup on failure
                await this.restoreBackup();
                throw error;
            }
        });
    }

    private async backupCurrentInstallation(): Promise<void> {
        const backupDir = path.join(this.context.globalStorageUri.fsPath, 'llm-kernel-backup');
        
        try {
            // Remove old backup if exists
            await fs.rm(backupDir, { recursive: true, force: true });
            
            // Copy current installation to backup
            await this.copyDirectory(this.kernelDir, backupDir);
        } catch (error) {
            this.outputChannel.appendLine(`Backup warning: ${error}`);
        }
    }

    private async restoreBackup(): Promise<void> {
        const backupDir = path.join(this.context.globalStorageUri.fsPath, 'llm-kernel-backup');
        
        try {
            await fs.rm(this.kernelDir, { recursive: true, force: true });
            await this.copyDirectory(backupDir, this.kernelDir);
            this.outputChannel.appendLine('Restored from backup successfully');
        } catch (error) {
            this.outputChannel.appendLine(`Failed to restore backup: ${error}`);
        }
    }

    private async copyDirectory(source: string, destination: string): Promise<void> {
        await fs.mkdir(destination, { recursive: true });
        
        const entries = await fs.readdir(source, { withFileTypes: true });
        
        for (const entry of entries) {
            const sourcePath = path.join(source, entry.name);
            const destPath = path.join(destination, entry.name);
            
            if (entry.isDirectory()) {
                await this.copyDirectory(sourcePath, destPath);
            } else {
                await fs.copyFile(sourcePath, destPath);
            }
        }
    }
}