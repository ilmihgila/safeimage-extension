/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';
import * as tfconv from '@tensorflow/tfjs-converter';

var EMBEDDING_NODES = 'module_apply_default/MobilenetV1/Logits/global_pool'
const TOPK_PREDICTIONS = 3;
const IMAGENET_CLASSES = {
    0: 'nude',
    1: 'safe',
    2: 'sexy'
};


const MODEL_DIR = './inimodelya/model.json';

var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function () { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function () { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var MobileNet = /** @class */ (function () {
    /**
     * Infer through MobileNet. This does standard ImageNet pre-processing before
     * inferring through the model. This method returns named activations as well
     * as softmax logits.
     *
     * @param input un-preprocessed input Array.
     * @return The softmax logits.
     */
    function hmz(inputMin, inputMax) {
        if (inputMin === void 0) { inputMin = -1; }
        if (inputMax === void 0) { inputMax = 1; }
        this.inputMin = inputMin;
        this.inputMax = inputMax;
        this.normalizationConstant = (inputMax - inputMin) / 255.0;
        this.modelUrl = MODEL_DIR;
    }
    hmz.prototype.load = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, url, _b, result;
            var _this = this;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        if (!this.modelUrl) return [3 /*break*/, 2];
                        _a = this;
                        return [4 /*yield*/, tfconv.loadGraphModel(this.modelUrl)];
                    case 1:
                        _a.model = _c.sent();
                        return [3 /*break*/, 4];
                    case 3:
                        _b.model = _c.sent();
                        _c.label = 4;
                    case 4:
                        result = tf.tidy(function () { return _this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])); });
                        return [4 /*yield*/, result.data()];
                    case 5:
                        _c.sent();
                        result.dispose();
                        return [2 /*return*/];
                }
            });
        });
    };


    hmz.prototype.infer = function (img, embedding) {
        var _this = this;
        if (embedding === void 0) { embedding = false; }
        return tf.tidy(function () {
            if (!(img instanceof tf.Tensor)) {
                img = tf.browser.fromPixels(img);
            }
            // Normalize the image from [0, 255] to [inputMin, inputMax].
            var normalized = tf.add(tf.mul(tf.cast(img, 'float32'), _this.normalizationConstant), _this.inputMin);
            // Resize the image to
            var resized = normalized;
            if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
                var alignCorners = true;
                resized = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
            }
            // Reshape so we can pass it to predict.
            var batched = tf.reshape(resized, [-1, IMAGE_SIZE, IMAGE_SIZE, 3]);
            var result;
            if (embedding) {
                var embeddingName = EMBEDDING_NODES;
                var internal = _this.model.execute(batched, embeddingName);
                result = tf.squeeze(internal, [1, 2]);
                console.log('embedding');
            }
            else {
                var logits1001 = _this.model.predict(batched);
                // Remove the very first logit (background noise).
                result = tf.slice(logits1001, [0, 0], [-1, 3]);
                console.log('not embedded');
            }
            return result;
        });
    };
    /**
     * Classifies an image from the 1000 ImageNet classes returning a map of
     * the most likely class names to their probability.
     *
     * @param img The image to classify. Can be a tensor or a DOM element image,
     * video, or canvas.
     * @param topk How many top values to use. Defaults to 3.
     */
    hmz.prototype.classify = function (img, topk) {
        if (topk === void 0) { topk = 3; }
        return __awaiter(this, void 0, void 0, function () {
            var logits, classes;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        logits = this.infer(img);
                        return [4 /*yield*/, getTopKClasses(logits, topk)];
                    case 1:
                        classes = _a.sent();
                        logits.dispose();
                        return [2 /*return*/, classes];
                }
            });
        });
    };
    return hmz;
}());

function getTopKClasses(logits, topK) {
    return __awaiter(this, void 0, void 0, function () {
        var softmax, values, valuesAndIndices, i, topkValues, topkIndices, i, topClassesAndProbs, i;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    softmax = tf.softmax(logits);
                    return [4 /*yield*/, softmax.data()];
                case 1:
                    values = _a.sent();
                    softmax.dispose();
                    valuesAndIndices = [];
                    for (i = 0; i < values.length; i++) {
                        valuesAndIndices.push({ value: values[i], index: i });
                    }
                    valuesAndIndices.sort(function (a, b) {
                        return b.value - a.value;
                    });
                    topkValues = new Float32Array(topK);
                    topkIndices = new Int32Array(topK);
                    for (i = 0; i < topK; i++) {
                        topkValues[i] = valuesAndIndices[i].value;
                        topkIndices[i] = valuesAndIndices[i].index;
                    }
                    topClassesAndProbs = [];
                    for (i = 0; i < topkIndices.length; i++) {
                        topClassesAndProbs.push({
                            className: IMAGENET_CLASSES[topkIndices[i]],
                            probability: topkValues[i]
                        });
                    }
                    return [2 /*return*/, topClassesAndProbs];
            }
        });
    });
}
// Size of the image expected by mobilenet.
const IMAGE_SIZE = 224;
const FIVE_SECONDS_IN_MS = 5000;

chrome.webNavigation.onDOMContentLoaded.addListener(() => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
            const tab = tabs[0];
            detectImages(tab)
                .then((srcUrls) => {
                    if (srcUrls.length > 0) {
                        return Promise.all(srcUrls.map((srcUrl) => detectTHEFLICKINIMAGES(srcUrl, tab)));
                    }
                })
                .catch((error) => {
                    console.error(error);
                });
        }
    });
});


function detectImages(tab) {
    return new Promise((resolve, reject) => {
        const message = { action: 'DETECT_IMAGES' };
        if (tab && tab.id) {
            chrome.tabs.sendMessage(tab.id, message, (resp) => {
                if (chrome.runtime.lastError) {
                    console.error(chrome.runtime.lastError);
                    reject(new Error('Failed to send message to content script'));
                } else {
                    const srcUrls = resp.srcUrls || [];
                    resolve(srcUrls);
                }
            });
        } else {
            console.error('Tab information is missing or invalid.');
            reject(new Error('Invalid tab information'));
        }
    });
}

function detectTHEFLICKINIMAGES(srcUrl, tab) {
    return new Promise((resolve, reject) => {
        const message = { action: 'DA_IMAGUS_BRADER', url: srcUrl };
        chrome.tabs.sendMessage(tab.id, message, (resp) => {
            if (!resp || !resp.rawImageData) {
                console.error(
                    'Failed to get image data. ' +
                    'The image might be too small or failed to load. ' +
                    'See console logs for errors.');
                reject(new Error('Failed to get image data'));
                return;
            } else {
                const imageData = new ImageData(
                    Uint8ClampedArray.from(resp.rawImageData), resp.width, resp.height);
                imageClassifier.analyzeImage(imageData, srcUrl, tab.id);
                resolve();
            }
        });
    });
}

class ImageClassifier {
    constructor() {
        this.loadModel();
    }

    /**
     * Loads mobilenet from URL and keeps a reference to it in the object.
     */
    async loadModel() {
        console.log('Loading model...');
        const startTime = performance.now();
        try {
            this.model = new MobileNet()
            await this.model.load();
            // Warms up the model by causing intermediate tensor values
            // to be built and pushed to GPU.
            tf.tidy(() => {
                this.model.classify(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
            });
            const totalTime = Math.floor(performance.now() - startTime);
            console.log(`Model loaded and initialized in ${totalTime} ms...`);
        } catch (e) {
            console.error('Unable to load model', e);
        }
    }

    /**
     * Triggers the model to make a prediction on the image referenced by the
     * image data. After a successful prediction a IMAGE_CLICK_PROCESSED message
     * when complete, for the content.js script to hear and update the DOM with
     * the results of the prediction.
     *
     * @param {ImageData} imageData ImageData of the image to analyze.
     * @param {string} url url of image to analyze.
     * @param {number} tabId which tab the request comes from.
     */
    async analyzeImage(imageData, url, tabId) {
        if (!tabId) {
            console.error('No tab.  No prediction.');
            return;
        }
        if (!this.model) {
            console.log('Waiting for model to load...');
            setTimeout(
                () => { this.analyzeImage(imageData, url, tabId) }, FIVE_SECONDS_IN_MS);
            return;
        }
        console.log('Predicting...');
        const startTime = performance.now();
        const predictions = await this.model.classify(imageData, TOPK_PREDICTIONS);
        const totalTime = performance.now() - startTime;
        console.log(`Done in ${totalTime.toFixed(1)} ms `);
        const message = { action: 'IMAGE_CLICK_PROCESSED', url, predictions };
        console.log(predictions);
        chrome.tabs.sendMessage(tabId, message);
    }
}

const imageClassifier = new ImageClassifier();