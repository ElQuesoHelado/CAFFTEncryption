#pragma once
#include "dft_traits.hpp"

#include <kfr/io/audiofile.hpp>
#include <kfr/io/file.hpp>

namespace dft {
struct Audio {
  kfr::univector2d<kfr::fbase> signals; // Stereo
  kfr::audio_format_and_length fmt;
  Traits traits;

  double time() const { return fmt.length / fmt.samplerate; }

  static Audio MakeFromWAV(const std::string &filename) {
    using namespace kfr;
    audio_reader_wav<fbase> reader{open_file_for_reading(filename)};
    const auto &fmt = reader.format();
    return {reader.read_channels(), fmt, Traits(fmt.samplerate, fmt.channels)};
  }
};
} // namespace dft
