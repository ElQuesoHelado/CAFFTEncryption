#pragma once
#include <vector>

namespace dft {
   struct Audio;

   Audio encrypt(const Audio& audio, const std::vector<unsigned>& data);
   std::vector<unsigned> decrypt(const Audio& audio, size_t size);
} // namespace dft