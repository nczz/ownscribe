import Accelerate
import ScreenCaptureKit
import CoreMedia
import AVFAudio
import CoreGraphics
import Foundation
import AppKit
import CoreAudio
import AudioToolbox
import IOKit.pwr_mgt

// MARK: - Constants

/// Minimum peak amplitude to consider microphone audio "loud" (silence timeout).
private let kMicLoudThreshold: Float = 1e-2
/// Minimum peak amplitude to consider system audio "loud" (silence timeout).
private let kSystemLoudThreshold: Float = 1e-4
/// System audio capture and merge output sample rate (mono float32).
private let kSystemAudioSampleRate: Double = 24000
/// Increment when the Python wrapper requires helper behavior older binaries lack.
private let kHelperProtocolVersion = 2

/// Compute peak amplitude across all channels of float audio data.
func computePeakLevel(in channelData: UnsafePointer<UnsafeMutablePointer<Float>>,
                      channels: Int, frames: Int) -> Float {
    guard channels > 0, frames > 0 else { return 0 }
    var peak: Float = 0
    for channel in 0..<channels {
        var channelPeak: Float = 0
        vDSP_maxmgv(
            channelData[channel],
            1,
            &channelPeak,
            vDSP_Length(frames))
        peak = max(peak, channelPeak)
    }
    return peak
}

// MARK: - Mic Capture via AVAudioEngine

class MicCapture {
    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private(set) var startHostTime: UInt64 = 0

    // Fixed processing format of the output file. Every tap buffer is normalised to
    // this before writing, so a mid-stream format change cannot break the writes the
    // way a fixed AVAudioFile did before (Meet flipping the shared input into
    // 3-channel voice-processing mode made every write fail with -50, killing the mic).
    private var fileFormat: AVAudioFormat?
    private var audioConverter: AVAudioConverter?
    private var conversionBuffer: AVAudioPCMBuffer?

    // Render-thread timing used to preserve the mic timeline across route changes.
    // handleConfigChange sets the gap flag only after removeTap has quiesced this thread.
    private var fillGapBeforeNextBuffer = false
    private var lastAudioCallbackUptime: UInt64?

    // Set when a write/convert failure has been logged; reset on the next successful write
    // so a persistent failure logs once instead of once per tap buffer (~10+/sec). Touched
    // only from writeBuffer on the tap (render) thread, so it needs no locking.
    private var loggedWriteError = false

    // Remembered so an explicitly named device can be re-applied after a route change.
    private var micDeviceName: String?

    // Re-bind the input + reinstall the tap when the audio route changes. The token is
    // kept so stop() can deregister it. isStopping guards a notification already queued
    // on .main behind stop(): the observer is delivered on .main and stop() runs on
    // .main (the SIGINT/SIGTERM sources), so they serialise and the flag alone is enough.
    private var configObserver: NSObjectProtocol?
    private var isStopping = false
    private var tapInstalled = false
    private var reconfigureGeneration = 0

    // Mute support — guarded by os_unfair_lock (tap callback is on AVAudioEngine thread)
    private var _isMuted = false
    private var _muteLock = os_unfair_lock_s()

    // Level tracking for silence timeout (mirrors SystemAudioCapture pattern)
    private var _lastLoudTime: UInt64 = DispatchTime.now().uptimeNanoseconds
    private var _lastLoudTimeLock = os_unfair_lock_s()

    var lastLoudTime: UInt64 {
        os_unfair_lock_lock(&_lastLoudTimeLock)
        defer { os_unfair_lock_unlock(&_lastLoudTimeLock) }
        return _lastLoudTime
    }

    var isMuted: Bool {
        os_unfair_lock_lock(&_muteLock)
        defer { os_unfair_lock_unlock(&_muteLock) }
        return _isMuted
    }

    func toggleMute() {
        os_unfair_lock_lock(&_muteLock)
        _isMuted.toggle()
        let muted = _isMuted
        os_unfair_lock_unlock(&_muteLock)
        fputs(muted ? "[MIC_MUTED]\n" : "[MIC_UNMUTED]\n", stderr)
    }

    func start(outputPath: String, deviceName: String?) throws {
        micDeviceName = deviceName
        try applyInputDevice()

        let format = engine.inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            throw MicError.noInputAvailable
        }

        let url = URL(fileURLWithPath: outputPath)
        let file = try AVAudioFile(forWriting: url,
                                   settings: format.settings,
                                   commonFormat: .pcmFormatFloat32,
                                   interleaved: true)
        audioFile = file
        fileFormat = file.processingFormat

        guard installMicTap() else {
            throw MicError.noInputAvailable
        }

        do {
            try engine.start()
        } catch {
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
                tapInstalled = false
            }
            audioFile = nil
            fileFormat = nil
            throw error
        }

        // Earbuds plugged in mid-recording, or a conferencing app reconfiguring the
        // shared input format, posts this notification (the engine stops itself first).
        // Rebind + restart so the mic keeps recording instead of silently dying.
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: .main
        ) { [weak self] _ in self?.handleConfigChange() }
        fputs("Recording microphone audio to \(outputPath)...\n", stderr)
    }

    /// Re-apply the explicitly named input device, if one was requested. When no device
    /// was named the engine follows the system default input — which is exactly what
    /// makes "switch to earbuds mid-recording" work, so we deliberately do nothing.
    private func applyInputDevice() throws {
        guard let name = micDeviceName else { return }
        // audioUnit can be nil while the node is in flux during a route change; treat
        // that as "cannot set device" rather than force-unwrapping into a crash.
        guard let audioUnit = engine.inputNode.audioUnit else {
            throw MicError.cannotSetDevice(name)
        }
        let deviceID = try findInputDevice(named: name)
        var id = deviceID
        let err = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0,
            &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        if err != noErr {
            throw MicError.cannotSetDevice(name)
        }
    }

    /// Install the capture tap using the input node's current format. Returns false if
    /// no input is available yet (sample rate 0), e.g. mid-route-change.
    @discardableResult
    private func installMicTap() -> Bool {
        guard !tapInstalled else { return true }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { return false }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, time in
            let callbackUptime = DispatchTime.now().uptimeNanoseconds
            guard let self else { return }
            if self.startHostTime == 0 {
                self.startHostTime = time.hostTime
            }
            let muted = self.isMuted
            if muted, let channelData = buffer.floatChannelData {
                let channels = Int(buffer.format.channelCount)
                let frames = Int(buffer.frameLength)
                for ch in 0..<channels {
                    memset(channelData[ch], 0, frames * MemoryLayout<Float>.size)
                }
            }
            // Track peak level for silence timeout (muted mic = silence)
            if !muted, let channelData = buffer.floatChannelData {
                let peak = computePeakLevel(in: channelData,
                                     channels: Int(buffer.format.channelCount),
                                     frames: Int(buffer.frameLength))
                if peak > kMicLoudThreshold {
                    os_unfair_lock_lock(&self._lastLoudTimeLock)
                    self._lastLoudTime = DispatchTime.now().uptimeNanoseconds
                    os_unfair_lock_unlock(&self._lastLoudTimeLock)
                }
            }
            self.writeBuffer(buffer, callbackUptime: callbackUptime)
        }
        tapInstalled = true
        return true
    }

    /// Write a tap buffer to the file, normalizing it to the file's fixed format.
    /// The conversion buffer is reused because conferencing apps can leave the input
    /// in a different format for hours; allocating one buffer per tap is unnecessary.
    private func writeBuffer(_ buffer: AVAudioPCMBuffer, callbackUptime: UInt64) {
        guard let audioFile, let fileFormat else { return }
        do {
            let outputBuffer: AVAudioPCMBuffer
            if buffer.format == fileFormat {
                outputBuffer = buffer
            } else {
                if audioConverter?.inputFormat != buffer.format
                    || audioConverter?.outputFormat != fileFormat {
                    audioConverter = AVAudioConverter(from: buffer.format, to: fileFormat)
                    conversionBuffer = nil
                }
                guard let converter = audioConverter else {
                    logWriteErrorOnce("Mic convert error: converter unavailable for format \(buffer.format)")
                    return
                }

                let ratio = fileFormat.sampleRate / buffer.format.sampleRate
                let requiredCapacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio)) + 16
                let converted: AVAudioPCMBuffer
                if let existing = conversionBuffer,
                   existing.format == fileFormat,
                   existing.frameCapacity >= requiredCapacity {
                    converted = existing
                } else {
                    guard let allocated = AVAudioPCMBuffer(
                        pcmFormat: fileFormat,
                        frameCapacity: requiredCapacity
                    ) else {
                        logWriteErrorOnce("Mic convert error: output buffer allocation failed")
                        return
                    }
                    conversionBuffer = allocated
                    converted = allocated
                }
                converted.frameLength = 0

                var fed = false
                var conversionError: NSError?
                let status = converter.convert(to: converted, error: &conversionError) { _, outStatus in
                    if fed {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    fed = true
                    outStatus.pointee = .haveData
                    return buffer
                }
                if status == .error || conversionError != nil {
                    logWriteErrorOnce(
                        "Mic convert error: \(conversionError?.description ?? "unknown")")
                    return
                }
                guard converted.frameLength > 0 else { return }
                outputBuffer = converted
            }

            try writeRecoveryGapIfNeeded(
                callbackUptime: callbackUptime,
                currentFrameCount: outputBuffer.frameLength,
                to: audioFile,
                sampleRate: fileFormat.sampleRate)
            try audioFile.write(from: outputBuffer)
            lastAudioCallbackUptime = callbackUptime
            loggedWriteError = false
        } catch {
            logWriteErrorOnce("Mic write error: \(error)")
        }
    }

    private func writeRecoveryGapIfNeeded(
        callbackUptime: UInt64,
        currentFrameCount: AVAudioFrameCount,
        to audioFile: AVAudioFile,
        sampleRate: Double
    ) throws {
        guard fillGapBeforeNextBuffer else { return }
        fillGapBeforeNextBuffer = false
        guard let previousUptime = lastAudioCallbackUptime, callbackUptime > previousUptime else { return }

        let elapsedFrames = Int64(
            (Double(callbackUptime - previousUptime) * sampleRate / 1_000_000_000.0).rounded())
        let gapFrames = max(0, elapsedFrames - Int64(currentFrameCount))
        guard gapFrames > 0 else { return }

        let capacity = AVAudioFrameCount(min(gapFrames, 8192))
        guard let silence = AVAudioPCMBuffer(
            pcmFormat: audioFile.processingFormat,
            frameCapacity: capacity
        ) else {
            logWriteErrorOnce("Mic gap fill error: output buffer allocation failed")
            return
        }
        silence.frameLength = capacity
        for audioBuffer in UnsafeMutableAudioBufferListPointer(silence.mutableAudioBufferList) {
            if let data = audioBuffer.mData {
                memset(data, 0, Int(audioBuffer.mDataByteSize))
            }
        }

        var remaining = gapFrames
        while remaining > 0 {
            silence.frameLength = AVAudioFrameCount(min(remaining, Int64(capacity)))
            try audioFile.write(from: silence)
            remaining -= Int64(silence.frameLength)
        }
    }

    /// Log a recurring write/convert failure only once until the next successful write, so a
    /// persistent bad state (e.g. an unconvertible format) doesn't flood stderr at the tap
    /// rate. Reset by writeBuffer on every successful write. Render-thread only — no locking.
    private func logWriteErrorOnce(_ message: String) {
        guard !loggedWriteError else { return }
        loggedWriteError = true
        fputs(message + "\n", stderr)
    }

    /// Rebind the input and reinstall the tap after an audio route change. Notifications
    /// and retries run on .main; a generation invalidates retries from older changes.
    private func handleConfigChange() {
        guard !isStopping else { return }
        reconfigureGeneration += 1
        let generation = reconfigureGeneration
        fputs("Mic reconfigure: audio route changed; rebinding microphone input.\n", stderr)
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        engine.stop()
        if lastAudioCallbackUptime != nil {
            fillGapBeforeNextBuffer = true
        }
        do {
            try applyInputDevice()
        } catch {
            fputs("Mic reconfigure: \(error) — falling back to default input.\n", stderr)
        }
        reinstallAndStart(generation: generation)
    }

    /// Retry for as long as recording remains active. Audio devices can take longer
    /// than a fixed five-second window to return after Bluetooth or dock changes.
    private func reinstallAndStart(generation: Int, attempt: Int = 0) {
        guard !isStopping, generation == reconfigureGeneration else { return }

        guard installMicTap() else {
            scheduleReinstall(generation: generation, attempt: attempt)
            return
        }
        do {
            try engine.start()
        } catch {
            if tapInstalled {
                engine.inputNode.removeTap(onBus: 0)
                tapInstalled = false
            }
            scheduleReinstall(generation: generation, attempt: attempt)
        }
    }

    private func scheduleReinstall(generation: Int, attempt: Int) {
        if attempt == 20 {
            fputs("Mic reconfigure: input still unavailable; continuing to retry.\n", stderr)
        }
        let delay = attempt < 20 ? 0.25 : 1.0
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.reinstallAndStart(generation: generation, attempt: attempt + 1)
        }
    }

    func stop() {
        isStopping = true
        reconfigureGeneration += 1
        if let token = configObserver {
            NotificationCenter.default.removeObserver(token)
            configObserver = nil
        }
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        engine.stop()
        audioFile = nil
        fileFormat = nil
        audioConverter = nil
        conversionBuffer = nil
    }

    enum MicError: Error, CustomStringConvertible {
        case deviceNotFound(String)
        case cannotSetDevice(String)
        case noInputAvailable

        var description: String {
            switch self {
            case .deviceNotFound(let n): return "Input device not found: \(n)"
            case .cannotSetDevice(let n): return "Cannot set input device: \(n)"
            case .noInputAvailable: return "No audio input available (sample rate is 0)"
            }
        }
    }
}

/// Find an input audio device by name, returning its AudioDeviceID.
func findInputDevice(named name: String) throws -> AudioDeviceID {
    var propAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)

    var dataSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                    &propAddress, 0, nil, &dataSize)
    let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var devices = [AudioDeviceID](repeating: 0, count: deviceCount)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &propAddress, 0, nil, &dataSize, &devices)

    for deviceID in devices {
        // Get device name
        var nameAddr = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cfNameRef: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddr, 0, nil, &nameSize, &cfNameRef)
        let deviceName = cfNameRef?.takeRetainedValue() as String? ?? "(unknown)"

        // Check input channel count
        var inputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var bufSize: UInt32 = 0
        AudioObjectGetPropertyDataSize(deviceID, &inputAddr, 0, nil, &bufSize)
        if bufSize > 0 {
            let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { bufferList.deallocate() }
            AudioObjectGetPropertyData(deviceID, &inputAddr, 0, nil, &bufSize, bufferList)
            let inputChannels = UnsafeMutableAudioBufferListPointer(bufferList)
                .reduce(0) { $0 + Int($1.mNumberChannels) }
            if inputChannels > 0 && deviceName.lowercased().contains(name.lowercased()) {
                return deviceID
            }
        }
    }
    throw MicCapture.MicError.deviceNotFound(name)
}

// MARK: - System Audio Capture via ScreenCaptureKit

class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate, SCContentSharingPickerObserver {
    private var stream: SCStream?
    private var audioFile: AVAudioFile?
    private var audioConverter: AVAudioConverter?
    private var audioInputBuffer: AVAudioPCMBuffer?
    private var audioConversionBuffer: AVAudioPCMBuffer?
    private var loggedSystemWriteError = false
    private let captureQueue = DispatchQueue(label: "com.ownscribe.audioCapture", qos: .userInitiated)

    // Accessed only on captureQueue. A restarted SCStream has no samples for the
    // interruption, so the first recovered buffer must restore that wall-clock gap.
    private var fillGapBeforeNextBuffer = false
    private var lastAudioCallbackUptime: UInt64?
    private var acceptsAudioSamples = false

    private let outputPath: String

    // Timestamp for sync alignment
    private(set) var startHostTime: UInt64 = 0

    // Silence detection
    private var peakLevel: Float = 0.0
    private var totalFrames: Int64 = 0
    private var silenceChecked: Bool = false

    // Silence timeout auto-stop
    var silenceTimeout: TimeInterval = 0  // seconds; 0 = disabled
    var onSilenceTimeout: (() -> Void)?
    var onStopRequested: (() -> Void)?
    var micCapture: MicCapture?  // checked by silence timer
    private var lastLoudTime: UInt64 = DispatchTime.now().uptimeNanoseconds
    private var lastLoudTimeLock = os_unfair_lock_s()
    private var silenceTimer: DispatchSourceTimer?

    // Power assertion to prevent display sleep during capture
    private var powerAssertionID: IOPMAssertionID = IOPMAssertionID(kIOPMNullAssertionID)

    // Capture lifecycle. ScreenCaptureKit stops a stream with noCaptureSource when
    // the selected display disappears (for example during a display topology change).
    // In all-audio mode we can bind a new stream to the current main display while
    // keeping the WAV file open, so microphone recording and system audio can continue.
    private var isStopping = false
    private var sourceUnavailable = false
    private var recoveryTask: Task<Void, Never>?

    // Picker continuation
    private var startContinuation: CheckedContinuation<Void, Error>?

    init(outputPath: String) {
        self.outputPath = outputPath
        super.init()
    }

    var captureModeAll: Bool = false

    func start() async throws {
        acquireDisplaySleepAssertion()
        if captureModeAll {
            let filter = try await waitForAllCaptureFilter()
            try await self.beginCapture(with: filter)
        } else {
            let picker = SCContentSharingPicker.shared
            var pickerConfig = SCContentSharingPickerConfiguration()
            pickerConfig.allowedPickerModes = [.singleWindow, .singleDisplay, .singleApplication]
            picker.defaultConfiguration = pickerConfig
            picker.add(self)
            picker.isActive = true
            picker.present()

            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                self.startContinuation = continuation
            }
        }
    }

    private func acquireDisplaySleepAssertion() {
        guard powerAssertionID == IOPMAssertionID(kIOPMNullAssertionID) else { return }
        let result = IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventUserIdleDisplaySleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "ownscribe is recording audio" as CFString,
            &powerAssertionID)
        if result != kIOReturnSuccess {
            fputs("Warning: could not prevent display sleep (IOKit error \(result)).\n", stderr)
            powerAssertionID = IOPMAssertionID(kIOPMNullAssertionID)
        }
    }

    /// Use the main display instead of the first unordered shareable display. The
    /// audio filter covers all applications; the display only anchors ScreenCaptureKit.
    private func makeAllCaptureFilter() async throws -> SCContentFilter {
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
        guard !content.displays.isEmpty else {
            throw CaptureError.noDisplay
        }
        let mainDisplayID = CGMainDisplayID()
        let display = content.displays.first(where: { $0.displayID == mainDisplayID }) ?? content.displays[0]
        return SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
    }

    private func waitForAllCaptureFilter() async throws -> SCContentFilter {
        var reportedWait = false
        while !isStopping && !Task.isCancelled {
            do {
                let filter = try await makeAllCaptureFilter()
                sourceUnavailable = false
                return filter
            } catch {
                guard isRetryableRecoveryError(error) else { throw error }
                sourceUnavailable = true
                if !reportedWait {
                    reportedWait = true
                    fputs("System audio source unavailable at startup; waiting to connect...\n", stderr)
                }
                try await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
        throw CancellationError()
    }

    // MARK: - SCContentSharingPickerObserver

    func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter, for stream: SCStream?) {
        Task { @MainActor [weak self] in
            guard let self, !self.isStopping else { return }
            let recovering = self.sourceUnavailable
            do {
                try await self.beginCapture(with: filter, recovering: recovering)
                self.sourceUnavailable = false
                self.startContinuation?.resume()
                self.startContinuation = nil
            } catch {
                if let continuation = self.startContinuation {
                    continuation.resume(throwing: error)
                    self.startContinuation = nil
                } else {
                    self.sourceUnavailable = true
                    fputs("[SILENCE_WARNING] Could not restart system audio capture: \(error)\n", stderr)
                }
            }
        }
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.sourceUnavailable {
                fputs(
                    "[SILENCE_WARNING] Replacement source selection cancelled; microphone recording continues.\n",
                    stderr)
            } else if let onStopRequested = self.onStopRequested {
                onStopRequested()
            } else {
                self.startContinuation?.resume(throwing: CaptureError.pickerCancelled)
                self.startContinuation = nil
            }
        }
    }

    func contentSharingPickerStartDidFailWithError(_ error: Error) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.sourceUnavailable {
                fputs("[SILENCE_WARNING] Could not show replacement source picker: \(error)\n", stderr)
            } else {
                self.startContinuation?.resume(throwing: error)
                self.startContinuation = nil
            }
        }
    }

    // MARK: - Begin Capture

    private func beginCapture(with filter: SCContentFilter, recovering: Bool = false) async throws {
        // Configure stream (audio only, minimal video)
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = Int(kSystemAudioSampleRate)
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false

        // Keep the same file open when a lost display source is reconnected. Reopening
        // AVAudioFile here would truncate everything captured before the interruption.
        if audioFile == nil {
            let fileFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: kSystemAudioSampleRate,
                channels: 1,
                interleaved: true
            )!
            audioFile = try AVAudioFile(
                forWriting: URL(fileURLWithPath: outputPath),
                settings: fileFormat.settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: true
            )
        }

        // Create and start stream
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: captureQueue)

        // Initialize last-loud time only for the initial stream. An interruption is
        // real silence and must not reset the configured silence timeout.
        if silenceTimeout > 0 && totalFrames == 0 {
            lastLoudTime = DispatchTime.now().uptimeNanoseconds
        }

        captureQueue.sync {
            acceptsAudioSamples = true
            if recovering {
                fillGapBeforeNextBuffer = true
            }
        }
        try await stream.startCapture()
        if isStopping || Task.isCancelled {
            try? await stream.stopCapture()
            throw CancellationError()
        }
        self.stream = stream


        if recovering {
            fputs("System audio capture reconnected.\n", stderr)
        } else {
            fputs("Recording system audio to \(outputPath)... Press Ctrl+C to stop.\n", stderr)
        }

        // Start silence timeout timer if configured.
        // Checks every 1s whether both system audio and mic (if active) have been
        // quiet longer than silenceTimeout. Uses the most recent "loud" timestamp
        // from either source so that activity on either channel prevents auto-stop.
        if silenceTimeout > 0 && silenceTimer == nil {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 1, repeating: 1.0)
            timer.setEventHandler { [weak self] in
                guard let self else { return }
                let now = DispatchTime.now().uptimeNanoseconds
                os_unfair_lock_lock(&self.lastLoudTimeLock)
                var effectiveLastLoud = self.lastLoudTime
                os_unfair_lock_unlock(&self.lastLoudTimeLock)
                // If mic is active, use the more recent of the two
                if let mic = self.micCapture {
                    let micLastLoud = mic.lastLoudTime
                    if micLastLoud > effectiveLastLoud {
                        effectiveLastLoud = micLastLoud
                    }
                }
                guard now >= effectiveLastLoud else { return }
                let elapsed = Double(now - effectiveLastLoud) / 1_000_000_000.0
                if elapsed > self.silenceTimeout {
                    fputs("[SILENCE_TIMEOUT]\n", stderr)
                    self.silenceTimer?.cancel()
                    self.silenceTimer = nil
                    self.onSilenceTimeout?()
                }
            }
            timer.resume()
            silenceTimer = timer
        }
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, acceptsAudioSamples, let audioFile else { return }
        let sampleCount = CMSampleBufferGetNumSamples(sampleBuffer)
        guard sampleCount > 0 else { return }

        let callbackUptime = DispatchTime.now().uptimeNanoseconds
        let callbackHostTime = mach_absolute_time()
        guard let formatDesc = sampleBuffer.formatDescription,
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc),
              let sampleFormat = AVAudioFormat(streamDescription: asbd) else { return }

        let inputFrameCount = AVAudioFrameCount(sampleCount)
        let inputBuffer: AVAudioPCMBuffer
        if let existing = audioInputBuffer,
           existing.format == sampleFormat,
           existing.frameCapacity >= inputFrameCount {
            inputBuffer = existing
        } else {
            guard let allocated = AVAudioPCMBuffer(
                pcmFormat: sampleFormat,
                frameCapacity: inputFrameCount
            ) else { return }
            audioInputBuffer = allocated
            inputBuffer = allocated
        }
        inputBuffer.frameLength = inputFrameCount

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(inputFrameCount),
            into: inputBuffer.mutableAudioBufferList
        )
        guard status == noErr else { return }

        do {
            guard let outputBuffer = try convertedSystemBuffer(inputBuffer, to: audioFile.processingFormat) else {
                return
            }
            let outputFrameCount = outputBuffer.frameLength
            try writeRecoveryGapIfNeeded(
                callbackUptime: callbackUptime,
                currentFrameCount: outputFrameCount,
                to: audioFile)
            try audioFile.write(from: outputBuffer)

            lastAudioCallbackUptime = callbackUptime
            totalFrames += Int64(outputFrameCount)
            if startHostTime == 0 {
                startHostTime = callbackHostTime
            }

            let bufferPeak: Float = outputBuffer.floatChannelData.map {
                computePeakLevel(
                    in: $0,
                    channels: Int(outputBuffer.format.channelCount),
                    frames: Int(outputFrameCount))
            } ?? 0.0
            if bufferPeak > peakLevel {
                peakLevel = bufferPeak
            }
            if bufferPeak > kSystemLoudThreshold {
                os_unfair_lock_lock(&lastLoudTimeLock)
                lastLoudTime = callbackUptime
                os_unfair_lock_unlock(&lastLoudTimeLock)
            }

            if !silenceChecked && Double(totalFrames) > kSystemAudioSampleRate * 3 {
                silenceChecked = true
                if peakLevel < 1e-6 {
                    if micCapture != nil {
                        fputs(
                            "[SILENCE_WARNING] System audio is silent; microphone recording continues. "
                            + "Play system audio and check Screen & System Audio Recording permission.\n",
                            stderr)
                    } else {
                        fputs(
                            "[SILENCE_WARNING] Audio data received but peak level is near zero "
                            + "(\(peakLevel)). Audio may be silent.\n",
                            stderr)
                        fputs(
                            "Check: System Settings > Privacy & Security > Screen & System Audio Recording "
                            + "— enable your terminal app.\n",
                            stderr)
                    }
                }
            }
            loggedSystemWriteError = false
        } catch {
            if !loggedSystemWriteError {
                loggedSystemWriteError = true
                fputs("System audio write error: \(error)\n", stderr)
            }
        }
    }

    private func convertedSystemBuffer(
        _ inputBuffer: AVAudioPCMBuffer,
        to outputFormat: AVAudioFormat
    ) throws -> AVAudioPCMBuffer? {
        if inputBuffer.format == outputFormat {
            return inputBuffer
        }

        if audioConverter?.inputFormat != inputBuffer.format
            || audioConverter?.outputFormat != outputFormat {
            audioConverter = AVAudioConverter(from: inputBuffer.format, to: outputFormat)
            audioConversionBuffer = nil
        }
        guard let converter = audioConverter else {
            throw CaptureError.cannotConvertAudio
        }

        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let requiredCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio)) + 16
        let outputBuffer: AVAudioPCMBuffer
        if let existing = audioConversionBuffer,
           existing.format == outputFormat,
           existing.frameCapacity >= requiredCapacity {
            outputBuffer = existing
        } else {
            guard let allocated = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: requiredCapacity
            ) else {
                throw CaptureError.cannotAllocateAudioBuffer
            }
            audioConversionBuffer = allocated
            outputBuffer = allocated
        }
        outputBuffer.frameLength = 0
        try converter.convert(to: outputBuffer, from: inputBuffer)
        return outputBuffer.frameLength > 0 ? outputBuffer : nil
    }

    private func writeRecoveryGapIfNeeded(
        callbackUptime: UInt64,
        currentFrameCount: AVAudioFrameCount,
        to audioFile: AVAudioFile
    ) throws {
        guard fillGapBeforeNextBuffer else { return }
        fillGapBeforeNextBuffer = false
        guard let previousUptime = lastAudioCallbackUptime, callbackUptime > previousUptime else { return }

        // Callback timestamps are near the end of each buffer. Subtract this buffer
        // from the elapsed time so ordinary callback latency is not inserted as silence.
        let elapsedNanos = callbackUptime - previousUptime
        let elapsedFrames = Int64(
            (Double(elapsedNanos) * kSystemAudioSampleRate / 1_000_000_000.0).rounded())
        let gapFrames = max(0, elapsedFrames - Int64(currentFrameCount))
        guard gapFrames > 0 else { return }

        try writeSilence(frames: gapFrames, to: audioFile)
        totalFrames += gapFrames
    }

    private func writeSilence(frames: Int64, to audioFile: AVAudioFile) throws {
        let capacity = AVAudioFrameCount(min(frames, 8192))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: capacity) else {
            return
        }
        buffer.frameLength = capacity
        for audioBuffer in UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList) {
            if let data = audioBuffer.mData {
                memset(data, 0, Int(audioBuffer.mDataByteSize))
            }
        }

        var remaining = frames
        while remaining > 0 {
            buffer.frameLength = AVAudioFrameCount(min(remaining, Int64(capacity)))
            try audioFile.write(from: buffer)
            remaining -= Int64(buffer.frameLength)
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { [weak self] in
            self?.handleStreamStop(stream, error: error)
        }
    }

    private func handleStreamStop(_ stoppedStream: SCStream, error: Error) {
        guard !isStopping, stream === stoppedStream else { return }
        stream = nil

        let nsError = error as NSError
        guard nsError.domain == SCStreamErrorDomain,
              let code = SCStreamError.Code(rawValue: nsError.code) else {
            sourceUnavailable = true
            fputs("[SILENCE_WARNING] System audio capture stopped: \(error)\n", stderr)
            return
        }

        switch code {
        case .userStopped:
            fputs("System audio capture stopped by user.\n", stderr)
            onStopRequested?()

        case .noCaptureSource,
             .noDisplayList,
             .noWindowList,
             .failedApplicationConnectionInvalid,
             .failedApplicationConnectionInterrupted,
             .systemStoppedStream:
            sourceUnavailable = true
            if captureModeAll {
                startSourceRecovery()
            } else {
                fputs("Selected system audio source disappeared; choose a replacement source.\n", stderr)
                SCContentSharingPicker.shared.present()
            }

        default:
            sourceUnavailable = true
            fputs("[SILENCE_WARNING] System audio capture stopped: \(error)\n", stderr)
        }
    }

    /// Re-query shareable displays until one is available. Display sleep/disconnect can
    /// leave ScreenCaptureKit without a source temporarily; microphone capture continues
    /// independently while this task waits.
    private func startSourceRecovery() {
        guard !isStopping, recoveryTask == nil else { return }

        sourceUnavailable = true
        fputs("System audio source became unavailable; waiting to reconnect...\n", stderr)

        recoveryTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled && !self.isStopping {
                do {
                    let filter = try await self.makeAllCaptureFilter()
                    guard !Task.isCancelled && !self.isStopping else { return }
                    try await self.beginCapture(with: filter, recovering: true)
                    self.sourceUnavailable = false
                    self.recoveryTask = nil
                    return
                } catch is CancellationError {
                    return
                } catch {
                    guard self.isRetryableRecoveryError(error) else {
                        self.recoveryTask = nil
                        fputs("[SILENCE_WARNING] System audio recovery failed: \(error)\n", stderr)
                        return
                    }
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                }
            }
        }
    }

    private func isRetryableRecoveryError(_ error: Error) -> Bool {
        if let captureError = error as? CaptureError, case .noDisplay = captureError {
            return true
        }
        let nsError = error as NSError
        guard nsError.domain == SCStreamErrorDomain,
              let code = SCStreamError.Code(rawValue: nsError.code) else {
            return false
        }
        switch code {
        case .noCaptureSource,
             .noDisplayList,
             .noWindowList,
             .failedApplicationConnectionInvalid,
             .failedApplicationConnectionInterrupted,
             .systemStoppedStream,
             .failedToStart,
             .failedToStartAudioCapture:
            return true
        default:
            return false
        }
    }

    // MARK: - Stop

    func stop() {
        guard !isStopping else { return }
        isStopping = true
        recoveryTask?.cancel()
        recoveryTask = nil
        if !captureModeAll {
            SCContentSharingPicker.shared.remove(self)
        }

        silenceTimer?.cancel()
        silenceTimer = nil

        if powerAssertionID != IOPMAssertionID(kIOPMNullAssertionID) {
            IOPMAssertionRelease(powerAssertionID)
            powerAssertionID = IOPMAssertionID(kIOPMNullAssertionID)
        }

        let sem = DispatchSemaphore(value: 0)
        Task.detached { [stream] in
            try? await stream?.stopCapture()
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 2)

        // Stop accepting callbacks before closing the file. stopCapture normally
        // quiesces output, but the timeout path must also be race-free.
        captureQueue.sync {
            acceptsAudioSamples = false
            audioFile = nil
            audioInputBuffer = nil
            audioConverter = nil
            audioConversionBuffer = nil
        }

        let seconds = Double(totalFrames) / kSystemAudioSampleRate
        fputs("Saved \(outputPath) (\(String(format: "%.1f", seconds)) seconds, peak=\(String(format: "%.6f", peakLevel)))\n", stderr)
        if sourceUnavailable {
            fputs(
                "[SILENCE_WARNING] System audio was unavailable when recording stopped; "
                + "part of the recording may contain microphone audio only.\n",
                stderr)
        } else if totalFrames > 0 && peakLevel < 1e-6 {
            fputs(
                "[SILENCE_WARNING] Recording appears silent. "
                + "Check Screen & System Audio Recording permission.\n",
                stderr)
        }
    }

    enum CaptureError: Error, CustomStringConvertible {
        case cannotOpenFile(String)
        case noDisplay
        case pickerCancelled
        case cannotConvertAudio
        case cannotAllocateAudioBuffer

        var description: String {
            switch self {
            case .cannotOpenFile(let p): return "Cannot open file: \(p)"
            case .noDisplay: return "No display found"
            case .pickerCancelled: return "Content picker cancelled"
            case .cannotConvertAudio: return "Cannot create system audio converter"
            case .cannotAllocateAudioBuffer: return "Cannot allocate audio conversion buffer"
            }
        }
    }
}

// MARK: - List apps

func listAudioApps() {
    print("Running apps:")
    for app in NSWorkspace.shared.runningApplications {
        if app.activationPolicy == .regular, let name = app.localizedName {
            print("  PID \(app.processIdentifier): \(name)")
        }
    }
}

// MARK: - List input devices

func listInputDevices() {
    // Get default input device
    var defaultAddr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var defaultDevice: AudioDeviceID = 0
    var defaultSize = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &defaultAddr, 0, nil, &defaultSize, &defaultDevice)

    // Enumerate all devices
    var propAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)

    var dataSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                    &propAddress, 0, nil, &dataSize)
    let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var devices = [AudioDeviceID](repeating: 0, count: deviceCount)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &propAddress, 0, nil, &dataSize, &devices)

    print("Input devices:")
    for deviceID in devices {
        // Get device name
        var nameAddr = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cfNameRef: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddr, 0, nil, &nameSize, &cfNameRef)
        let name = cfNameRef?.takeRetainedValue() as String? ?? "(unknown)"

        // Check input channel count
        var inputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var bufSize: UInt32 = 0
        AudioObjectGetPropertyDataSize(deviceID, &inputAddr, 0, nil, &bufSize)
        guard bufSize > 0 else { continue }

        let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        defer { bufferList.deallocate() }
        AudioObjectGetPropertyData(deviceID, &inputAddr, 0, nil, &bufSize, bufferList)
        let inputChannels = UnsafeMutableAudioBufferListPointer(bufferList)
            .reduce(0) { $0 + Int($1.mNumberChannels) }

        if inputChannels > 0 {
            let suffix = (deviceID == defaultDevice) ? " (default)" : ""
            print("  \(name)\(suffix)")
        }
    }
}

// MARK: - Merge audio files with timestamp alignment

enum AudioMergeError: Error, CustomStringConvertible {
    case cannotAllocateBuffer(String)
    case cannotCreateConverter
    case conversionFailed(String)

    var description: String {
        switch self {
        case .cannotAllocateBuffer(let name): return "Cannot allocate \(name) audio buffer"
        case .cannotCreateConverter: return "Cannot create microphone audio converter"
        case .conversionFailed(let detail): return "Microphone conversion failed: \(detail)"
        }
    }
}

func mergeAudioFiles(systemPath: String, micPath: String,
                     systemStartHostTime: UInt64, micStartHostTime: UInt64,
                     outputPath: String) throws {
    // A standard WAV file header (RIFF + fmt + data chunk header) is 44 bytes.
    // Files at or below this size contain no audio frames.
    let wavHeaderSize = 44
    let fm = FileManager.default
    let systemFileSize = (try? fm.attributesOfItem(atPath: systemPath)[.size] as? Int) ?? 0
    let micFileSize = (try? fm.attributesOfItem(atPath: micPath)[.size] as? Int) ?? 0

    // Both empty — clean up temp files and let the caller handle it
    if systemFileSize <= wavHeaderSize && micFileSize <= wavHeaderSize {
        try? fm.removeItem(atPath: systemPath)
        try? fm.removeItem(atPath: micPath)
        return
    }

    // Mic empty but system has data — just rename system file
    if micFileSize <= wavHeaderSize && systemFileSize > wavHeaderSize {
        try? fm.removeItem(atPath: micPath)
        try fm.moveItem(atPath: systemPath, toPath: outputPath)
        fputs("Merged audio saved to \(outputPath) (system only, no mic audio)\n", stderr)
        return
    }

    // Open files — system is optional (may be empty when only mic captured audio)
    let systemFile: AVAudioFile? = systemFileSize > wavHeaderSize
        ? try AVAudioFile(forReading: URL(fileURLWithPath: systemPath)) : nil
    let micFile = try AVAudioFile(forReading: URL(fileURLWithPath: micPath))

    let outputSampleRate: Double = kSystemAudioSampleRate
    let outputChannels: AVAudioChannelCount = 1

    // Compute offset in seconds between the two start times using mach_timebase_info
    let offsetFrames: Int64
    if systemFile != nil {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let ticksToNanos = Double(timebase.numer) / Double(timebase.denom)
        let systemStartNanos = Double(systemStartHostTime) * ticksToNanos
        let micStartNanos = Double(micStartHostTime) * ticksToNanos
        let offsetSeconds = (micStartNanos - systemStartNanos) / 1_000_000_000.0
        offsetFrames = Int64(offsetSeconds * outputSampleRate)
    } else {
        offsetFrames = 0
    }
    let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: outputSampleRate,
                                      channels: outputChannels,
                                      interleaved: true)!
    let outputFile = try AVAudioFile(forWriting: URL(fileURLWithPath: outputPath),
                                      settings: outputFormat.settings,
                                      commonFormat: .pcmFormatFloat32,
                                      interleaved: true)

    let micFormat = micFile.processingFormat
    guard let micConverter = AVAudioConverter(from: micFormat, to: outputFormat) else {
        throw AudioMergeError.cannotCreateConverter
    }
    let micRate = micFormat.sampleRate

    let chunkSize: AVAudioFrameCount = 8192
    let systemLength = systemFile.map { Int64($0.length) } ?? 0
    let micLengthOutput = Int64(Double(micFile.length) * outputSampleRate / micRate)

    let systemEndFrame = (offsetFrames >= 0) ? systemLength : systemLength + (-offsetFrames)
    let micEndFrame = (offsetFrames >= 0) ? micLengthOutput + offsetFrames : micLengthOutput
    let totalOutputFrames = max(systemEndFrame, micEndFrame)

    let micOutputStart: Int64 = (offsetFrames >= 0) ? offsetFrames : 0
    let micOutputEnd: Int64 = micOutputStart + micLengthOutput

    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: chunkSize
    ) else {
        throw AudioMergeError.cannotAllocateBuffer("output")
    }
    guard let micOutputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: chunkSize
    ) else {
        throw AudioMergeError.cannotAllocateBuffer("microphone output")
    }
    var partialMicOutputBuffer: AVAudioPCMBuffer?

    let systemBuffer: AVAudioPCMBuffer?
    if let systemFile {
        guard let allocated = AVAudioPCMBuffer(
            pcmFormat: systemFile.processingFormat,
            frameCapacity: chunkSize
        ) else {
            throw AudioMergeError.cannotAllocateBuffer("system input")
        }
        systemBuffer = allocated
    } else {
        systemBuffer = nil
    }

    var micInputBuffer: AVAudioPCMBuffer?
    var micDone = false
    var outputFrame: Int64 = 0

    while outputFrame < totalOutputFrames {
        let framesToProcess = AVAudioFrameCount(min(Int64(chunkSize), totalOutputFrames - outputFrame))
        outputBuffer.frameLength = framesToProcess
        guard let outPtr = outputBuffer.floatChannelData?[0] else {
            throw AudioMergeError.cannotAllocateBuffer("output samples")
        }

        let totalSamples = Int(framesToProcess * outputChannels)
        memset(outPtr, 0, totalSamples * MemoryLayout<Float>.size)

        if let systemFile, let systemBuffer {
            let systemFrame = (offsetFrames >= 0) ? outputFrame : outputFrame + offsetFrames
            let writeOffset = max(0, -systemFrame)
            let readStart = max(0, systemFrame)
            if readStart < systemLength && writeOffset < Int64(framesToProcess) {
                let maxRead = Int64(framesToProcess) - writeOffset
                let readCount = AVAudioFrameCount(min(maxRead, systemLength - readStart))
                if readCount > 0 {
                    systemFile.framePosition = AVAudioFramePosition(readStart)
                    try systemFile.read(into: systemBuffer, frameCount: readCount)
                    guard let systemData = systemBuffer.floatChannelData else {
                        throw AudioMergeError.cannotAllocateBuffer("system samples")
                    }
                    let channelCount = Int(systemBuffer.format.channelCount)
                    let destination = outPtr.advanced(by: Int(writeOffset))
                    let frameCount = Int(systemBuffer.frameLength)
                    if channelCount == 1 {
                        vDSP_vadd(
                            systemData[0], 1,
                            destination, 1,
                            destination, 1,
                            vDSP_Length(frameCount))
                    } else {
                        for frame in 0..<frameCount {
                            var mix: Float = 0
                            for channel in 0..<channelCount {
                                mix += systemData[channel][frame]
                            }
                            destination[frame] += mix / Float(channelCount)
                        }
                    }
                }
            }
        }

        let chunkEnd = outputFrame + Int64(framesToProcess)
        if !micDone && chunkEnd > micOutputStart && outputFrame < micOutputEnd {
            let overlapStart = max(outputFrame, micOutputStart)
            let overlapEnd = min(chunkEnd, micOutputEnd)
            let micFramesNeeded = AVAudioFrameCount(overlapEnd - overlapStart)

            if micFramesNeeded > 0 {
                let convertedMicBuffer: AVAudioPCMBuffer
                if micFramesNeeded == chunkSize {
                    convertedMicBuffer = micOutputBuffer
                } else if let existing = partialMicOutputBuffer,
                          existing.frameCapacity == micFramesNeeded {
                    convertedMicBuffer = existing
                } else {
                    guard let allocated = AVAudioPCMBuffer(
                        pcmFormat: outputFormat,
                        frameCapacity: micFramesNeeded
                    ) else {
                        throw AudioMergeError.cannotAllocateBuffer("partial microphone output")
                    }
                    partialMicOutputBuffer = allocated
                    convertedMicBuffer = allocated
                }
                convertedMicBuffer.frameLength = 0
                var conversionError: NSError?
                var readError: Error?
                var allocationFailed = false
                let status = micConverter.convert(
                    to: convertedMicBuffer,
                    error: &conversionError
                ) { requestedPackets, outputStatus in
                    let remaining = AVAudioFrameCount(micFile.length - micFile.framePosition)
                    if remaining == 0 {
                        outputStatus.pointee = .endOfStream
                        return nil
                    }

                    let readCount = min(requestedPackets, remaining)
                    guard readCount > 0 else {
                        outputStatus.pointee = .noDataNow
                        return nil
                    }

                    let inputBuffer: AVAudioPCMBuffer
                    if let existing = micInputBuffer,
                       existing.frameCapacity >= readCount {
                        inputBuffer = existing
                    } else if let allocated = AVAudioPCMBuffer(
                        pcmFormat: micFormat,
                        frameCapacity: readCount
                    ) {
                        micInputBuffer = allocated
                        inputBuffer = allocated
                    } else {
                        allocationFailed = true
                        outputStatus.pointee = .endOfStream
                        return nil
                    }

                    do {
                        try micFile.read(into: inputBuffer, frameCount: readCount)
                        outputStatus.pointee = .haveData
                        return inputBuffer
                    } catch {
                        readError = error
                        outputStatus.pointee = .endOfStream
                        return nil
                    }
                }

                if allocationFailed {
                    throw AudioMergeError.cannotAllocateBuffer("microphone input")
                }
                if let readError {
                    throw AudioMergeError.conversionFailed(readError.localizedDescription)
                }
                if status == .error || conversionError != nil {
                    throw AudioMergeError.conversionFailed(
                        conversionError?.localizedDescription ?? "unknown converter error")
                }
                if status == .endOfStream {
                    micDone = true
                }

                if convertedMicBuffer.frameLength > 0 {
                    guard convertedMicBuffer.frameLength <= micFramesNeeded else {
                        throw AudioMergeError.conversionFailed(
                            "converter produced more frames than the current timeline region")
                    }
                    guard let source = convertedMicBuffer.floatChannelData?[0] else {
                        throw AudioMergeError.cannotAllocateBuffer("microphone output samples")
                    }
                    let destination = outPtr.advanced(by: Int(overlapStart - outputFrame))
                    vDSP_vadd(
                        source, 1,
                        destination, 1,
                        destination, 1,
                        vDSP_Length(convertedMicBuffer.frameLength))
                }
            }
        }

        var lowerBound: Float = -1
        var upperBound: Float = 1
        vDSP_vclip(
            outPtr, 1,
            &lowerBound, &upperBound,
            outPtr, 1,
            vDSP_Length(totalSamples))

        try outputFile.write(from: outputBuffer)
        outputFrame += Int64(framesToProcess)
    }

    // Delete temp files
    try? FileManager.default.removeItem(atPath: systemPath)
    try? FileManager.default.removeItem(atPath: micPath)

    let detail = systemFile == nil ? " (mic only, no system audio)" : ""
    fputs("Merged audio saved to \(outputPath)\(detail)\n", stderr)
}

// MARK: - Main

func printUsage() {
    fputs("""
    ownscribe-audio — system audio capture helper

    USAGE:
        ownscribe-audio capture --output FILE [--mic] [--mic-device NAME] [--silence-timeout N]
        ownscribe-audio list-apps
        ownscribe-audio list-devices
        ownscribe-audio version

    OPTIONS:
        --output, -o FILE    Output WAV file path (required for capture)
        --mic                Also capture microphone input
        --mic-device NAME    Use specific mic input device (implies --mic)
        --capture-mode-all   Capture all system audio without showing the source picker
        --silence-timeout N  Auto-stop after N seconds of silence (0 = disabled)
        --help, -h           Show this help

    SUBCOMMANDS:
        capture              Record audio to a WAV file
        list-apps            Show running applications
        list-devices         Show available audio input devices
        version              Print the helper protocol version

    """, stderr)
}

func main() {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        printUsage()
        exit(1)
    }

    let command = args[1]

    switch command {
    case "version":
        print(kHelperProtocolVersion)

    case "list-apps":
        listAudioApps()

    case "list-devices":
        listInputDevices()

    case "capture":
        // Initialize NSApplication so the picker GUI can render
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)

        var outputPath: String?
        var enableMic = false
        var micDeviceName: String?
        var captureModeAll = false
        var silenceTimeout: TimeInterval = 0

        var i = 2
        while i < args.count {
            switch args[i] {
            case "--output", "-o":
                i += 1
                guard i < args.count else {
                    fputs("Error: --output requires a file path\n", stderr)
                    exit(1)
                }
                outputPath = args[i]
            case "--capture-mode-all":
                captureModeAll = true
            case "--mic":
                enableMic = true
            case "--mic-device":
                i += 1
                guard i < args.count else {
                    fputs("Error: --mic-device requires a device name\n", stderr)
                    exit(1)
                }
                micDeviceName = args[i]
                enableMic = true  // --mic-device implies --mic
            case "--silence-timeout":
                i += 1
                guard i < args.count, let val = TimeInterval(args[i]) else {
                    fputs("Error: --silence-timeout requires a number of seconds\n", stderr)
                    exit(1)
                }
                if val < 0 {
                    fputs("Error: --silence-timeout must be zero (disabled) or a positive number of seconds\n", stderr)
                    exit(1)
                }
                silenceTimeout = val
            default:
                fputs("Unknown option: \(args[i])\n", stderr)
                printUsage()
                exit(1)
            }
            i += 1
        }

        guard let output = outputPath else {
            fputs("Error: --output is required\n", stderr)
            printUsage()
            exit(1)
        }

        // Determine paths: if mic enabled, use temp files then merge
        let systemPath = enableMic ? output + ".sys.tmp.wav" : output
        let micPath = output + ".mic.tmp.wav"

        let capture = SystemAudioCapture(outputPath: systemPath)
        capture.captureModeAll = captureModeAll
        capture.silenceTimeout = silenceTimeout
        var micCapture: MicCapture?

        if enableMic {
            let mic = MicCapture()
            do {
                try mic.start(outputPath: micPath, deviceName: micDeviceName)
            } catch {
                fputs("Error starting mic capture: \(error)\n", stderr)
                exit(1)
            }
            micCapture = mic
            capture.micCapture = mic
        }

        // One finalizer owns capture shutdown and track merging. Signals, picker
        // cancellation, user-stopped streams, silence timeout, and startup failure all
        // use it so no path can strand the microphone temp file.
        var didFinishCapture = false
        var finalizationFailed = false
        let finishCapture: () -> Void = {
            guard !didFinishCapture else { return }
            didFinishCapture = true
            capture.stop()
            if let mic = micCapture {
                mic.stop()
                do {
                    try mergeAudioFiles(
                        systemPath: systemPath,
                        micPath: micPath,
                        systemStartHostTime: capture.startHostTime,
                        micStartHostTime: mic.startHostTime,
                        outputPath: output)
                } catch {
                    finalizationFailed = true
                    try? FileManager.default.removeItem(atPath: output)
                    fputs("Error merging audio: \(error)\n", stderr)
                }
            }
        }
        let shutdown: () -> Void = {
            finishCapture()
            exit(finalizationFailed ? 1 : 0)
        }

        capture.onSilenceTimeout = shutdown
        capture.onStopRequested = shutdown

        // Toggle mic mute on SIGUSR1 (sent by Python wrapper)
        var _sigusr1Source: DispatchSourceSignal?  // retained to keep source alive
        if let mic = micCapture {
            signal(SIGUSR1, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
            src.setEventHandler { mic.toggleMute() }
            src.resume()
            _sigusr1Source = src
        }
        _ = _sigusr1Source

        // Handle Ctrl+C gracefully
        let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        signal(SIGINT, SIG_IGN)
        sigintSource.setEventHandler { shutdown() }
        sigintSource.resume()

        let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        signal(SIGTERM, SIG_IGN)
        sigtermSource.setEventHandler { shutdown() }
        sigtermSource.resume()

        Task { @MainActor in
            do {
                try await capture.start()
            } catch {
                fputs("Error: \(error)\n", stderr)
                finishCapture()
                exit(1)
            }
        }

        app.run()

    case "--help", "-h":
        printUsage()

    default:
        fputs("Unknown command: \(command)\n", stderr)
        printUsage()
        exit(1)
    }
}

main()
