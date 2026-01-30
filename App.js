import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { Camera, useCameraDevice, useFrameProcessor, runAtTargetFps } from 'react-native-vision-camera';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { useAssets } from 'expo-asset';
import { Worklets, useSharedValue } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';

const MODEL_FILENAME = 'OptimizedIdCardOnDeviceTop1.onnx'; 
const INPUT_TENSOR_NAME = 'image_buffer'; 
const CONFIDENCE_THRESHOLD = 0.95;
const TARGET_FPS = 2; 


const { width: WINDOW_WIDTH, height: WINDOW_HEIGHT } = Dimensions.get('window');

// 0.83 and 1.62 parameters are for for id card rectangle
const CARD_WIDTH = WINDOW_WIDTH * 0.83; 
const CARD_HEIGHT = CARD_WIDTH * 1.62;



export default function App() {
  const [modelStatus, setModelStatus] = useState("Loading...");
  const [scanResult, setScanResult] = useState(null);
  const [latency, setLatency] = useState(0);

  const sessionRef = useRef(null);
  const isModelReady = useSharedValue(false);
  const isInferenceBusy = useSharedValue(false); 

  const { resize } = useResizePlugin();
  const [assets] = useAssets([require(`./assets/${MODEL_FILENAME}`)]);
  const device = useCameraDevice('back');

  useEffect(() => {
    if (assets?.[0]) {
      InferenceSession.create(assets[0].localUri)
        .then(session => {
          sessionRef.current = session;
          isModelReady.value = true;
          setModelStatus("✅ Ready");
        })
        .catch(e => setModelStatus("❌ Error"));
    }
  }, [assets]);

  const performInference = async (bgrArray) => {
    if (!sessionRef.current) return;
    try {
      const startTime = Date.now();
      const uint8Data = new Uint8Array(bgrArray);
      
      // We send the cropped data scaled to 320x240
      const inputTensor = new Tensor('uint8', uint8Data, [240, 320, 3]);
      const outputs = await sessionRef.current.run({ [INPUT_TENSOR_NAME]: inputTensor });
      
      const data = outputs[Object.keys(outputs)[0]].data; 
      const score = data[4];
      const clsId = data[5];

      if (score > CONFIDENCE_THRESHOLD) {
        setScanResult({ 
          label: Math.round(clsId) === 0 ? "FRONT" : "BACK", 
          score: (score * 100).toFixed(0) 
        });
      } else {
        setScanResult(null);
      }
      setLatency(Date.now() - startTime);
    } catch (e) {
      console.log("Inference Error:", e.message);
    } finally {
      isInferenceBusy.value = false;
    }
  };
  
  const runInferenceJS = Worklets.createRunOnJS(performInference);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    runAtTargetFps(TARGET_FPS, () => {
      'worklet';
      if (isInferenceBusy.value || !isModelReady.value) return;

      try {

        const cropW = frame.width * 0.83; 
        const cropH = cropW * 1.62;
        const cropX = (frame.width - cropW) / 2;
        const cropY = (frame.height - cropH) / 2;

        const resized = resize(frame, {
          crop: { x: cropX, y: cropY, width: cropW, height: cropH },
          scale: { width: 320, height: 240 },
          pixelFormat: 'bgr',
          dataType: 'uint8',
        });

        if (resized) {
          isInferenceBusy.value = true;
          runInferenceJS(Array.from(resized));
        }
      } catch (e) {
        console.log('Worklet Error:', e.message);
      }
    });
  }, []);

  if (!device) return null;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />

      {/* LANDSCAPE UI OVERLAY */}
      <View style={styles.uiLayer}>
        
        {/* The Scanning Rectangle  */}
        <View style={[
          styles.rectangle, 
          { borderColor: scanResult ? '#00FF00' : 'white' }
        ]}>
          <View style={styles.cornerTL} />
          <View style={styles.cornerTR} />
          <View style={styles.cornerBL} />
          <View style={styles.cornerBR} />
          
          {/* Label inside the box */}
          <Text style={[styles.hintText, { color: scanResult ? '#00FF00' : 'white' }]}>
            {scanResult ? `${scanResult.label} DETECTED` : "ALIGN ID CARD"}
          </Text>
        </View>

        {/* Bottom Data Info */}
        <View style={styles.footer}>
          <Text style={styles.infoText}>Status: {modelStatus} | {latency}ms</Text>
          {scanResult && (
            <Text style={styles.resultText}>
              IDENTIFIED: {scanResult.label} ({scanResult.score}%)
            </Text>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  
  uiLayer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },

  // Horizontal Rectangle
  rectangle: {
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    borderWidth: 1,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.05)',
  },

  // Corner Accents for a "Scanner" look
  cornerTL: { position: 'absolute', top: -2, left: -2, width: 30, height: 30, borderTopWidth: 5, borderLeftWidth: 5, borderColor: 'inherit', borderTopLeftRadius: 16 },
  cornerTR: { position: 'absolute', top: -2, right: -2, width: 30, height: 30, borderTopWidth: 5, borderRightWidth: 5, borderColor: 'inherit', borderTopRightRadius: 16 },
  cornerBL: { position: 'absolute', bottom: -2, left: -2, width: 30, height: 30, borderBottomWidth: 5, borderLeftWidth: 5, borderColor: 'inherit', borderBottomLeftRadius: 16 },
  cornerBR: { position: 'absolute', bottom: -2, right: -2, width: 30, height: 30, borderBottomWidth: 5, borderRightWidth: 5, borderColor: 'inherit', borderBottomRightRadius: 16 },

  hintText: {
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 2,
    textShadowColor: 'black',
    textShadowRadius: 4,
  },

  footer: {
    position: 'absolute',
    bottom: 60,
    alignItems: 'center',
  },
  infoText: { color: '#ccc', fontSize: 12, marginBottom: 10 },
  resultText: { 
    color: '#00FF00', 
    fontSize: 22, 
    fontWeight: 'bold', 
    backgroundColor: 'rgba(0,0,0,0.8)', 
    padding: 10, 
    borderRadius: 8 
  },
});