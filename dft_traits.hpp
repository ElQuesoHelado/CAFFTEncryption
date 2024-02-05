#pragma once
#include <cmath>
#include <stdexcept>

namespace dft {
class Traits {
  const unsigned m_samples;
  const double m_resolution;
  const double m_timePerSamples;
  const unsigned m_channels;

  static unsigned clp2(unsigned x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
  }

public:
  Traits(unsigned samplerate, const unsigned channels) : m_samples{clp2(samplerate)},
                                                         m_resolution{samplerate / (double)m_samples},
                                                         m_timePerSamples{1.0 / m_resolution},
                                                         m_channels{channels} {
    if (samplerate < 40000)
      throw std::runtime_error("required a minimum of 40kHz to use this encryption");
  }

  auto preferredSamples() const { return m_samples; }
  auto resolution() const { return m_resolution; }
  auto timePerSamples() const { return m_timePerSamples; }

  unsigned freqToIndex(double freq) const {
    return std::round(freq / m_resolution);
  }

  double minimumTimeForSize(size_t size) const {
    return std::ceil(size / (double)maximumNElementsPerTime()) * timePerSamples();
  }

  //
  unsigned maximumNElementsPerTime() const {
    return (freqToIndex(20.0) + preferredSamples() / 2 - freqToIndex(16000.0)) * m_channels /*stereo*/;
  }
};
} // namespace dft
