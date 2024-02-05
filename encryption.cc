#include "encryption.hpp"

#include "dft_audio.hpp"

#include <format>
#include <kfr/dft.hpp>
#include <kfr/dft/cache.hpp>
#include <stdexcept>

using namespace kfr;

static inline fbase sign(const fbase &x) {
  return std::signbit(x) ? -1.0 : 1.0;
}

namespace dft {
Audio encrypt(const Audio &audio, const std::vector<unsigned> &data) {
  static_assert(std::same_as<fbase, double>, "floating-point must be double");

  if (auto minimum_time = audio.traits.minimumTimeForSize(data.size()); audio.time() < minimum_time)
    throw std::runtime_error(std::format("duration of audio ({:.02})s is lower that required ({:.02}s)", audio.time(), minimum_time));

  const auto samples = audio.traits.preferredSamples();
  const auto idx20 = audio.traits.freqToIndex(20.0);
  const auto idx16k = audio.traits.freqToIndex(16000.0);

  univector2d<fbase> res;
  dft_plan_real_ptr<fbase> plan{dft_cache::instance().getreal(ctype<fbase>, samples)};
  if (plan->fmt != dft_pack_format::CCs)
    throw std::runtime_error("format must be CCs");

  univector<u8> temp(plan->temp_size);
  univector<complex<fbase>> out(samples / 2 + 1);
  univector<fbase> out_inv(samples);
  size_t encrypted{};

  for (const auto &signal : audio.signals) {
    auto &res_signal = res.emplace_back(signal.size());
    auto it = res_signal.begin();
    size_t segment{};
    for (; segment + samples < signal.size() && encrypted != data.size(); segment += samples) {
      plan->execute(out, (univector<fbase>)signal.slice(segment), temp);
      out /= samples; // Normalizando DFT

      for (size_t j = 1; j < out.size() - 1 && encrypted != data.size(); ++j) {
        if (j == idx20 + 1) {
          j = idx16k;
          continue;
        }

        auto &x = out[j];
        // Aplicando lo del paper
        const auto r = real(x);
        const auto img = imag(x);
        const auto abs_r = abs(r);
        const auto abs_i = abs(img);
        const fbase v = data.at(encrypted++) / fbase(0xFFFFFF);
        if (abs_r >= abs_i) {
          x = {r, sign(img) * abs_r * v};
        } else {
          x = {sign(r) * abs_i * v, img};
        }
      }

      plan->execute(out_inv, out, temp);
      // restableciendo el segmento
      it = std::ranges::copy(out_inv, it).out;
    }

    // copiando los sobrantes
    std::ranges::copy(signal.slice(segment), it);
  }

  return {res, audio.fmt, audio.traits};
}

std::vector<unsigned> decrypt(const Audio &audio, size_t size) {
  const auto samples = audio.traits.preferredSamples();
  const auto idx20 = audio.traits.freqToIndex(20.0);
  const auto idx16k = audio.traits.freqToIndex(16000.0);

  dft_plan_real_ptr<fbase> plan{dft_cache::instance().getreal(ctype<fbase>, samples)};
  if (plan->fmt != dft_pack_format::CCs)
    throw std::runtime_error("format must be CCs");
  univector<u8> temp(plan->temp_size);
  univector<complex<fbase>> out(samples / 2 + 1);
  std::vector<unsigned> res;
  size_t encrypted{};

  for (const auto &signal : audio.signals) {
    for (size_t segment{}; segment + samples < signal.size() && encrypted != size; segment += samples) {
      plan->execute(out, (univector<fbase>)signal.slice(segment, samples), temp);
      out /= samples; // Normalizando

      for (size_t i = 1; i < out.size() - 1 && encrypted != size; ++i) {
        if (i == idx20 + 1) {
          i = idx16k;
          continue;
        }

        const auto &x = out[i];
        // Método del paper
        const auto abs_r = abs(x.real());
        const auto abs_i = abs(x.imag());
        const unsigned v = round(min(abs_r, abs_i) / max(abs_r, abs_i) * 0xFFFFFF);
        res.push_back(v);
        ++encrypted;
      }
    }
  }

  return res;
}
} // namespace dft
