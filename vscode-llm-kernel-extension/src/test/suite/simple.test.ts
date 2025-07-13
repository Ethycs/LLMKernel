import * as assert from 'assert';

// Simple test to verify test infrastructure is working
suite('Simple Test Suite', () => {
    
    test('Basic assertion test', () => {
        assert.strictEqual(1 + 1, 2);
    });

    test('String test', () => {
        const message = 'Hello, VS Code!';
        assert.ok(message.includes('VS Code'));
    });

    test('Async test', async () => {
        const promise = new Promise<string>((resolve) => {
            setTimeout(() => resolve('done'), 100);
        });
        
        const result = await promise;
        assert.strictEqual(result, 'done');
    });

    test('Array test', () => {
        const arr = [1, 2, 3, 4, 5];
        assert.strictEqual(arr.length, 5);
        assert.ok(arr.includes(3));
    });
});