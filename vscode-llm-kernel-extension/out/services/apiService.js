"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ApiService = void 0;
const axios_1 = __importDefault(require("axios"));
class ApiService {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    sendRequest(endpoint, data) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const response = yield axios_1.default.post(`${this.baseUrl}/${endpoint}`, data);
                return response.data;
            }
            catch (error) {
                console.error('API request failed:', error);
                throw error;
            }
        });
    }
    getRequest(endpoint) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const response = yield axios_1.default.get(`${this.baseUrl}/${endpoint}`);
                return response.data;
            }
            catch (error) {
                console.error('API request failed:', error);
                throw error;
            }
        });
    }
    // Context management methods
    getCurrentContext() {
        return __awaiter(this, void 0, void 0, function* () {
            return this.getRequest('context/current');
        });
    }
    saveContext(name, data) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.sendRequest('context/save', { name, data });
        });
    }
    loadContext(name) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.getRequest(`context/load/${name}`);
        });
    }
    setContext(data) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.sendRequest('context/set', data);
        });
    }
    resetContext(keepHidden = false) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.sendRequest('context/reset', { keepHidden });
        });
    }
    getContextStatus() {
        return __awaiter(this, void 0, void 0, function* () {
            const response = yield this.getRequest('context/status');
            return response.status;
        });
    }
    // Kernel methods  
    sendKernelRequest(data) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.sendRequest('kernel/request', data);
        });
    }
}
exports.ApiService = ApiService;
