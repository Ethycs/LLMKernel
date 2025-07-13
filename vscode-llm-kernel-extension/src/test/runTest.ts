import * as path from 'path';
import { runTests, downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath } from '@vscode/test-electron';

async function main() {
    try {
        // The folder containing the Extension Manifest package.json
        const extensionDevelopmentPath = path.resolve(__dirname, '../../');

        // The path to the extension test runner script
        const extensionTestsPath = path.resolve(__dirname, './suite/index');

        console.log('🧪 Setting up VS Code Extension Tests...');
        console.log(`📁 Extension path: ${extensionDevelopmentPath}`);
        console.log(`🔧 Test path: ${extensionTestsPath}`);

        // WSL environment detection and handling
        if (process.env.WSL_DISTRO_NAME) {
            console.log('🐧 Detected WSL environment');
            
            // Download and unzip VS Code for testing
            console.log('⬇️ Downloading VS Code for testing...');
            const vscodeExecutablePath = await downloadAndUnzipVSCode('stable');
            console.log(`📦 VS Code downloaded to: ${vscodeExecutablePath}`);

            // Get CLI arguments for the downloaded VS Code
            const [cli, ...args] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath);

            // Run tests with downloaded VS Code
            await runTests({
                vscodeExecutablePath,
                extensionDevelopmentPath,
                extensionTestsPath,
                launchArgs: [
                    '--disable-extensions',
                    '--disable-workspace-trust',
                    '--skip-welcome',
                    '--skip-release-notes',
                    '--no-sandbox',
                    '--disable-gpu-sandbox'
                ]
            });
        } else {
            // Non-WSL environment - use standard approach
            console.log('💻 Running in standard environment');
            await runTests({
                extensionDevelopmentPath,
                extensionTestsPath,
                launchArgs: [
                    '--disable-extensions',
                    '--disable-workspace-trust'
                ]
            });
        }

        console.log('✅ All extension tests completed successfully!');
    } catch (err) {
        console.error('❌ Failed to run tests:', err);
        
        // Provide helpful debugging information
        console.log('\n🔍 Debugging Information:');
        console.log(`- WSL_DISTRO_NAME: ${process.env.WSL_DISTRO_NAME || 'Not set'}`);
        console.log(`- Extension path exists: ${require('fs').existsSync(path.resolve(__dirname, '../../'))}`);
        console.log(`- Test path exists: ${require('fs').existsSync(path.resolve(__dirname, './suite/index'))}`);
        
        console.log('\n💡 Troubleshooting:');
        console.log('1. Ensure extension compiles: npm run compile');
        console.log('2. Try running unit tests: node ./out/test/unitTest.js');
        console.log('3. Test extension manually: ./test-extension.sh');
        
        process.exit(1);
    }
}

main();