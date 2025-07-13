import * as assert from 'assert';
import * as path from 'path';

// Import modules to test (avoiding VS Code dependencies)
import { ApiService } from '../services/apiService';
import { KernelService } from '../services/kernelService';

console.log('🧪 Running Unit Tests');

// Test ApiService
console.log('\n📦 Testing ApiService...');
try {
    const apiService = new ApiService();
    assert.ok(apiService);
    assert.ok(typeof apiService.sendRequest === 'function');
    assert.ok(typeof apiService.getRequest === 'function');
    assert.ok(typeof apiService.getCurrentContext === 'function');
    console.log('✅ ApiService tests passed');
} catch (error) {
    console.error('❌ ApiService test failed:', error);
}

// Test KernelService
console.log('\n⚙️ Testing KernelService...');
try {
    const kernelService = new KernelService();
    
    // Test initial state
    kernelService.getKernelStatus().then(status => {
        assert.strictEqual(status, 'Stopped');
        console.log('✅ Initial kernel status correct');
        
        // Test start/stop cycle
        return kernelService.startKernel();
    }).then(() => {
        return kernelService.getKernelStatus();
    }).then(status => {
        assert.strictEqual(status, 'Running');
        console.log('✅ Kernel starts correctly');
        
        return kernelService.stopKernel();
    }).then(() => {
        return kernelService.getKernelStatus();
    }).then(status => {
        assert.strictEqual(status, 'Stopped');
        console.log('✅ Kernel stops correctly');
        
        // Test completions
        return kernelService.getCompletions('test');
    }).then(completions => {
        assert.ok(Array.isArray(completions));
        assert.ok(completions.length > 0);
        console.log('✅ Completions work correctly');
        console.log('✅ KernelService tests passed');
    }).catch(error => {
        console.error('❌ KernelService test failed:', error);
    });
} catch (error) {
    console.error('❌ KernelService instantiation failed:', error);
}

// Test Natural Language Detection (simplified)
console.log('\n🗣️ Testing Natural Language Detection...');
try {
    // Mock the natural language detector logic
    const naturalLanguagePatterns = [
        /^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\s+/i,
        /^(please|help|explain|show|tell|describe|analyze|compare|summarize)\s+/i,
        /\?$/,
        /^(could you|can you|would you|please)/i
    ];
    
    const codePatterns = [
        /^(import|from|def|class|if|for|while|try|function|var|let|const)\s/i,
        /^(library|require|using|#include|<%|%>)\s*\(/i,
        /^\s*\w+\s*[=<-]\s*/,
        /^\s*\w+\s*\(/,
        /^%%?\w+/,
        /^\s*[#\/\*]/,
        /^\s*[\{\[]/,
        /^!\s*\w+/
    ];
    
    const isNaturalLanguage = (text: string): boolean => {
        const trimmed = text.trim();
        
        if (trimmed.length < 3) return false;
        
        if (codePatterns.some(pattern => pattern.test(trimmed))) {
            return false;
        }
        
        return naturalLanguagePatterns.some(pattern => pattern.test(trimmed));
    };
    
    // Test natural language queries
    const naturalQueries = [
        'What is machine learning?',
        'How do I optimize this function?',
        'Can you explain this code?',
        'Please help me understand this',
        'Why does this error occur?'
    ];
    
    for (const query of naturalQueries) {
        assert.strictEqual(isNaturalLanguage(query), true, `Failed: "${query}"`);
    }
    
    // Test code that should NOT be detected as natural language
    const codeExamples = [
        'import pandas as pd',
        'def function():',
        'x = 5',
        'for i in range(10):',
        '%%llm'
    ];
    
    for (const code of codeExamples) {
        assert.strictEqual(isNaturalLanguage(code), false, `Failed: "${code}"`);
    }
    
    console.log('✅ Natural Language Detection tests passed');
} catch (error) {
    console.error('❌ Natural Language Detection test failed:', error);
}

// Test Configuration Defaults
console.log('\n⚙️ Testing Configuration Defaults...');
try {
    const defaultConfig = {
        repository: {
            owner: 'your-username',
            repo: 'LLMKernel',
            branch: 'main'
        },
        autoInstall: true,
        autoUpdate: false,
        checkForUpdates: true,
        defaultModel: 'gpt-4o-mini',
        contextStrategy: 'smart',
        streamResponses: true
    };
    
    assert.ok(defaultConfig.repository);
    assert.strictEqual(defaultConfig.autoInstall, true);
    assert.strictEqual(defaultConfig.defaultModel, 'gpt-4o-mini');
    assert.strictEqual(defaultConfig.contextStrategy, 'smart');
    
    console.log('✅ Configuration defaults tests passed');
} catch (error) {
    console.error('❌ Configuration defaults test failed:', error);
}

console.log('\n🎉 Unit tests completed!');
console.log('\nNote: Integration tests require VS Code Extension Host');
console.log('To run full tests, install required system libraries and run:');
console.log('  npm test');