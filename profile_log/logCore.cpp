#include "logCore.hpp"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <string>

#if ENABLE_PROFILE_LOG

#pragma warning(disable:4834)

void LogCore::ProfileLog::cacheInStep(float avgFPS, float timeElapsed) {
    //m_frameRates.push_back(avgFPS);
    m_frameRates.push_back(1000.f / timeElapsed);
    m_frameDurations.push_back(timeElapsed);
    for (size_t i = 0; i < m_cudaEvents.size(); ++i) {
        cudaEventSynchronize(m_cudaEvents[i]);
    }
    m_eventDurations.emplace_back();
    if (m_cudaEvents.size() > 0) {
        std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations.back();
        eventDurationMap.reserve(m_cudaEvents.size() - 1);
        for (size_t i = 0; i + 1 < m_cudaEvents.size(); ++i) {
            eventDurationMap[i] = 0.f;
            cudaEventElapsedTime(&eventDurationMap[i], m_cudaEvents[i], m_cudaEvents[i + 1]);
        }
    }
}

LogCore::ProfileLog& LogCore::ProfileLog::get() {
    static ProfileLog profileLog;
    return profileLog;
}

bool LogCore::ProfileLog::isRecordingEvent() const {
    return m_recording;
}

bool LogCore::ProfileLog::isProfiling() const {
    return m_profiling;
}

const std::string& LogCore::ProfileLog::getEndEventName() const {
    return m_endEventName;
}

void LogCore::ProfileLog::clearArgs() {
    m_args.empty();
    m_kwargs.empty();
}

void LogCore::ProfileLog::addArg(const std::string& arg) {
    m_args.push_back(arg);
}

void LogCore::ProfileLog::addKwArg(const std::string& key, int value) {
    if (m_kwargs.find(key) == m_kwargs.end()) {
        m_kwargs_keys.push_back(key);
    }
    m_kwargs[key] = value;
}

void LogCore::ProfileLog::initProfile(const std::string& prefix, const std::string& endEventName, int startFrame, int frameCount, int frameInterval, std::ios_base::openmode mode) {
    m_mode = mode;
    m_endEventName = endEventName;
    m_startFrame = startFrame;
    m_endFrame = startFrame + frameCount * frameInterval;

    //m_startTime = startTime;
    m_frameInterval = frameInterval;

    //m_dataSizeToCount = (m_endFrame - m_startFrame) / m_frameInterval;
    m_dataSizeToCount = frameCount;

    m_frameRates.reserve(m_dataSizeToCount);
    m_frameDurations.reserve(m_dataSizeToCount);
    m_eventDurations.reserve(m_dataSizeToCount);

    m_filePath = prefix;
    //for (const std::pair<std::string, int>& kw : m_kwargs) {
    //    m_filePath.append("," + kw.first + "_" + std::to_string(kw.second));
    for(const std::string& arg : m_kwargs_keys) {
        m_filePath.append("," + arg + "_" + std::to_string(m_kwargs[arg]));
    }
    for (const std::string& arg : m_args) {
        m_filePath.append("," + arg);
    }
    m_filePath.append(".csv");
    m_recording = true;
}

size_t LogCore::ProfileLog::registerEvent(const std::string& eventName) {
    auto it = m_cudaEventIndices.find(eventName);
    if (it != m_cudaEventIndices.end()) {
        return it->second;
    }
    size_t result = m_cudaEvents.size();
    m_cudaEventNames.push_back(eventName);
    m_cudaEventIndices[eventName] = result;
    m_cudaEvents.emplace_back();
    cudaEvent_t& newEvent = m_cudaEvents.back();
    cudaEventCreate(&newEvent);
    return result;
}

void LogCore::ProfileLog::recordEvent(const std::string& eventName) {
    size_t index = registerEvent(eventName);
    if (!m_recording) {
        return;
    }
    cudaEvent_t& event = m_cudaEvents[index];
    cudaEventRecord(event);
}

void LogCore::ProfileLog::step(float avgFPS, float timeStamp){
    float timeElapsed = timeStamp - m_timeStamp;
    m_timeStamp = timeStamp;
    m_profiling = false;

    if (m_frameCount >= m_startFrame && m_frameCount < m_endFrame) {
    //if(timeStamp >= m_startTime) {
    //    if (m_startFrame == -1) {
    //        m_startFrame = m_frameCount;
    //        m_endFrame = m_frameCount + m_dataSizeToCount * m_frameInterval;
    //    }
    //    if (m_frameCount < m_endFrame) {
            if (m_frameInterval <= 1 || (m_frameCount - m_startFrame) % m_frameInterval == 0) {
                cacheInStep(avgFPS, timeElapsed);
            }
            m_profiling = true;
    //    }
    }

    if(!m_profiling) {
        if (m_recording && m_frameCount >= m_endFrame) {
            writeToFile(m_mode);
            unregisterEvents();
        }
    }
    ++m_frameCount;
}
void LogCore::ProfileLog::writeToFile(std::ios_base::openmode mode) {
    if (m_eventDurations.size() == 0) {
        std::cout << "No event durations... Cancelling write profile log to file." << std::endl;
        return;
    }
    
    m_fout.open(m_filePath, mode);
    m_fout << ",fps,duration";
    m_fout << ",cudaTotal";
    for (size_t i = 0; i + 1 < m_cudaEventNames.size(); ++i) {
        const std::string& header = m_cudaEventNames[i];
        m_fout << ',' << header;
    }
    m_fout << std::endl;
    int frameCount = m_startFrame;
    
    double avgFPS = 0., avgDuration = 0., avgTotal = 0.;
    double maxFPS = 0., maxDuration = 0., maxTotal = 0.;
    double minFPS = std::numeric_limits<double>::max(), minDuration = std::numeric_limits<double>::max(), minTotal = std::numeric_limits<double>::max();

    std::unordered_map<size_t, double> avgEventDurations;
    std::unordered_map<size_t, double> maxEventDurations;
    std::unordered_map<size_t, double> minEventDurations;

    for (std::pair<const size_t, float>& eventPair : m_eventDurations[0]) {
        avgEventDurations[eventPair.first] = 0.;
        maxEventDurations[eventPair.first] = 0.;
        minEventDurations[eventPair.first] = std::numeric_limits<double>::max();
    }
    double invCount = 1. / m_eventDurations.size();

    std::stringstream ss;

    for (size_t i = 0; i < m_frameRates.size(); ++i) {
        double cudaTotalDuration = 0.;

        avgFPS += m_frameRates[i] * invCount;
        avgDuration += m_frameDurations[i] * invCount;

        maxFPS = std::max<double>(maxFPS, m_frameRates[i]);
        minFPS = std::min<double>(minFPS, m_frameRates[i]);

        maxDuration = std::max<double>(maxDuration, m_frameDurations[i]);
        minDuration = std::min<double>(minDuration, m_frameDurations[i]);

        //m_fout << frameCount << ',' << m_frameRates[i] << ',' << m_frameDurations[i];

        std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations[i];
        for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
            avgEventDurations[eventPair.first] += eventPair.second * invCount;
            maxEventDurations[eventPair.first] = std::max<double>(maxEventDurations[eventPair.first], eventPair.second);
            minEventDurations[eventPair.first] = std::min<double>(minEventDurations[eventPair.first], eventPair.second);
            cudaTotalDuration += eventPair.second;
            //m_fout << ',' << eventPair.second;
        }
        avgTotal += cudaTotalDuration * invCount;
        maxTotal = std::max(maxTotal, static_cast<double>(cudaTotalDuration));
        minTotal = std::min(minTotal, static_cast<double>(cudaTotalDuration));
        //m_fout << ',' << cudaTotalDuration << std::endl;
    }
    m_fout << "avg_" << m_frameRates.size() << ',' << avgFPS << ',' << avgDuration;
    m_fout << ',' << avgTotal;
    for (std::pair<const size_t, double>& eventPair : avgEventDurations) {
        m_fout << ',' << eventPair.second;
    }
    m_fout << std::endl;

    m_fout << "max_" << m_frameRates.size() << ',' << maxFPS << ',' << maxDuration;
    m_fout << ',' << maxTotal;
    for (std::pair<const size_t, double>& eventPair : maxEventDurations) {
        m_fout << ',' << eventPair.second;
    }
    m_fout << std::endl;

    m_fout << "min_" << m_frameRates.size() << ',' << minFPS << ',' << minDuration;
    m_fout << ',' << minTotal;
    for (std::pair<const size_t, double>& eventPair : minEventDurations) {
        m_fout << ',' << eventPair.second;
    }
    m_fout << std::endl;

    for (size_t i = 0; i < m_frameRates.size(); ++i) {
        double cudaTotalDuration = 0.;

        m_fout << frameCount << ',' << m_frameRates[i] << ',' << m_frameDurations[i];

        std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations[i];
        for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
            avgEventDurations[eventPair.first] += eventPair.second * invCount;
            cudaTotalDuration += eventPair.second;
            //m_fout << ',' << eventPair.second;
        }
        m_fout << ',' << cudaTotalDuration;
        for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
            m_fout << ',' << eventPair.second;
        }
        m_fout << std::endl;
        frameCount += m_frameInterval;
    }

    m_fout.close();

    std::cout << "Write profile at " << m_filePath << std::endl;
}

void LogCore::ProfileLog::unregisterEvents() {
    m_recording = false;
    m_cudaEventIndices.empty();
    m_cudaEventNames.empty();
    for (size_t i = 0; i < m_cudaEvents.size(); ++i) {
        cudaEventDestroy(m_cudaEvents[i]);
    }
    m_cudaEvents.empty();
}
#endif // ENABLE_PROFILE_LOG
