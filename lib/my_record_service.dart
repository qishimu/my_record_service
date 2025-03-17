import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import "dart:io";

class AsrManager {
  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.OfflineRecognizer? _nonstreamRecognizer;

  bool _isInitialized = false;

  Future<void> _init() async {
    if (_isInitialized) return;

    _vad = await initVad();
    _nonstreamRecognizer = await createNonstreamingRecognizer();
    _isInitialized = true;
  }

  bool _isSpeechDetected(Float32List samples) {
    if (_vad == null) return false;
    _vad!.acceptWaveform(samples);
    return _vad!.isDetected();
  }

  Future<String> _getTranscription(Float32List audioData) async {
    if (_nonstreamRecognizer == null) throw Exception("ASR not initialized");

    final nonstreamStream = _nonstreamRecognizer!.createStream();
    nonstreamStream.acceptWaveform(samples: audioData, sampleRate: 16000);
    _nonstreamRecognizer!.decode(nonstreamStream);

    final text = _nonstreamRecognizer!.getResult(nonstreamStream).text;
    nonstreamStream.free();
    return text;
  }

  // Voice Activity Detector（VAD）初始化
  Future<sherpa_onnx.VoiceActivityDetector> initVad() async =>
      sherpa_onnx.VoiceActivityDetector(
        config: sherpa_onnx.VadModelConfig(
          sileroVad: sherpa_onnx.SileroVadModelConfig(
            model: await copyAssetFile('assets/silero_vad.onnx'),
            minSilenceDuration: 0.25,
            minSpeechDuration: 0.5,
            maxSpeechDuration: 5.0,
          ),
          numThreads: 1,
          debug: true,
        ),
        bufferSizeInSeconds: 2.0,
      );

  // 离线语音识别器初始化
  Future<sherpa_onnx.OfflineRecognizer> createNonstreamingRecognizer() async {
    final modelConfig = await getNonstreamingModelConfig();
    final config = sherpa_onnx.OfflineRecognizerConfig(
      model: modelConfig,
      ruleFsts: '',
    );
    return sherpa_onnx.OfflineRecognizer(config);
  }

  // 离线模型配置
  Future<sherpa_onnx.OfflineModelConfig> getNonstreamingModelConfig() async {
    const modelDir = 'assets/sherpa-onnx-whisper-tiny.en';
    return sherpa_onnx.OfflineModelConfig(
      whisper: sherpa_onnx.OfflineWhisperModelConfig(
        encoder: await copyAssetFile('$modelDir/tiny.en-encoder.int8.onnx'),
        decoder: await copyAssetFile('$modelDir/tiny.en-decoder.int8.onnx'),
        tailPaddings: 2000,
      ),
      tokens: await copyAssetFile('$modelDir/tiny.en-tokens.txt'),
    );
  }

  // 复制文件到设备
  Future<String> copyAssetFile(String src, [String? dst]) async {
    final directory = await getApplicationDocumentsDirectory();
    dst ??= basename(src);
    final target = join(directory.path, dst);

    final targetFile = File(target);
    if (await targetFile.exists()) {
      return target;
    }

    final data = await rootBundle.load(src);
    final List<int> bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await targetFile.writeAsBytes(bytes);

    return target;
  }

  // 将字节数据转换为 Float32
  Float32List convertBytesToFloat32(Uint8List bytes, [endian = Endian.little]) {
    final values = Float32List(bytes.length ~/ 2);
    final data = ByteData.view(bytes.buffer);

    for (var i = 0; i < bytes.length; i += 2) {
      int short = data.getInt16(i, endian);
      values[i ~/ 2] = short / 32767.0;
    }

    return values;
  }

  void dispose() {
    _vad?.free();
    _nonstreamRecognizer?.free();
    _isInitialized = false;
  }
}
