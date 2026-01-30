const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);

// Allow Metro to bundle .onnx files correctly
config.resolver.assetExts.push('onnx');

module.exports = config;