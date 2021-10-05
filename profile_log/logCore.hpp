#pragma once

/*********************
* Start Profile Log *
*********************/

#define ENABLE_PROFILE_LOG 1//0

#if ENABLE_PROFILE_LOG
#include <cuda_runtime.h>
#include <fstream>
#include <unordered_map>
namespace LogCore {
class ProfileLog {
private:
    std::ofstream m_fout;
    std::string m_filePath;
    std::string m_endEventName;

    bool m_recording = false;
    bool m_profiling = false;

    std::vector<std::string> m_args;
    std::vector<std::string> m_kwargs_keys;
    std::unordered_map<std::string, int> m_kwargs;

    std::ios_base::openmode m_mode = std::ios_base::out;
    float m_timeStamp = 0.f;
    int m_frameCount = 0, m_dataSizeToCount = 2048;
    int m_startFrame = -1, m_endFrame = -1;
    float m_startTime = 2000.f;
    int m_frameInterval = 1;

    std::vector<float> m_frameRates;
    std::vector<float> m_frameDurations;
    std::vector<std::unordered_map<size_t, float>> m_eventDurations;

    std::unordered_map<std::string, size_t> m_cudaEventIndices;
    std::vector<std::string> m_cudaEventNames;
    std::vector<cudaEvent_t> m_cudaEvents;

private:
    void cacheInStep(float avgFPS, float timeElapsed);

public:
    static ProfileLog& get();
    bool isRecordingEvent() const;
    bool isProfiling() const;
    const std::string& getEndEventName() const;
    
    void clearArgs();
    void addArg(const std::string& arg);
    void addKwArg(const std::string& key, int value);

    void initProfile(const std::string& prefix, const std::string& endEventName = "end", int startFrame = 256, int frameCount = 2048, int frameInterval = 1, std::ios_base::openmode mode = std::ios_base::out);
    size_t registerEvent(const std::string& eventName);
    void recordEvent(const std::string& eventName);

    // Call before runCUDA, timeStamp in ms
    void step(float avgFPS, float timeStamp);

    void writeToFile(std::ios_base::openmode mode = std::ios_base::out);
    void unregisterEvents();
};
}

#define callCUDA_Profile(funcName) \
    utilityCore::ProfileLog::get().recordEvent(#funcName "-" + std::to_string(__LINE__)); \
    funcName

#define callCUDA_ProfileWithAlias(alias, funcName) \
    utilityCore::ProfileLog::get().recordEvent(#alias); \
    funcName

#define callCUDA_ProfileEnd() utilityCore::ProfileLog::get().recordEvent(utilityCore::ProfileLog::get().getEndEventName())

#else // ENABLE_PROFILE_LOG
#define callCUDA_Profile(funcName) funcName
#define callCUDA_ProfileWithAlias(alias, funcName) funcName
#define callCUDA_ProfileEnd()
#endif // ENABLE_PROFILE_LOG

/*******************
* End Profile Log *
*******************/
