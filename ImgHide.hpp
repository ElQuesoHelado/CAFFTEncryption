#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <list>
#include <ostream>
#include <sndfile-64.h>
#include <string>

#include <Magick++.h>
#include <opencv2/opencv.hpp>

#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>
#include <sndfile.hh>
#include <vector>

#include "dft_audio.hpp"
#include "encryption.hpp"
#include "kfr/base/conversion.hpp"
#include "kfr/cometa/string.hpp"

namespace {
std::vector<unsigned> mat_to_pixel(const cv::Mat &img) {
  std::vector<unsigned> pixeles;
  pixeles.reserve(img.rows * img.cols);

  for (size_t r{}; r < img.rows; ++r) {
    for (size_t c{}; c < img.cols; ++c) {
      auto pix = img.at<cv::Vec3b>(r, c);
      pixeles.push_back((pix[0] << 16) + (pix[1] << 8) + pix[2]);
    }
  }
  return pixeles;
}

cv::Mat pixel_to_mat(const std::vector<unsigned> &pixeles, size_t rows, size_t cols) {
  cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));

  auto it = pixeles.begin();
  for (size_t r{}; r < rows; ++r) {
    for (size_t c{}; c < cols; ++c) {
      img.at<cv::Vec3b>(r, c) = {(uchar)(*it >> 16), (uchar)(*it >> 8), (uchar)*it++};
      /* pixeles.push_back((pix[0] << 16) + (pix[1] << 8) + pix[2]); */
    }
  }
  return img;
}
} // namespace

namespace ImgHide {

void analizar_audio(const std::string &path) {
  dft::Audio audio{dft::Audio::MakeFromWAV(path)};
  size_t cap_frag_audio = audio.traits.maximumNElementsPerTime(),
         size_fft = audio.traits.preferredSamples(), fragmentos = audio.fmt.length / size_fft;
  std::cout << std::format("\n Audio:{} \n Usando un bloque de: {}, fragmentos: {}\n"
                           "Pixeles por bloque: {}, totales: {}\n"
                           "Dims img max por bloque: {}, total: {}",
                           path, size_fft, fragmentos, cap_frag_audio, cap_frag_audio * fragmentos,
                           std::sqrt(cap_frag_audio), std::sqrt(cap_frag_audio * fragmentos))
            << std::endl;
}

void save_audio(const dft::Audio &audio, const std::string &name) {
  using namespace kfr;
  SndfileHandle file;
  int channels = audio.fmt.channels;
  int srate = audio.fmt.samplerate;
  int format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE; // wav64

  file = SndfileHandle(name, SFM_WRITE, format, channels, srate);

  file.write(kfr::interleave(audio.signals).data(), audio.signals[0].size() * audio.fmt.channels);
}

/*
 * Esconde imagenes dentro de un directorio en un conjunto de imagenes,
 * las imagenes deben tener mismas dimensiones, solo RGB
 * @param path directorio de imagenes
 * @param path directorio de audios
 */
void hide(const std::string &img_dir_path, const std::string &audio_dir_path) {
  using namespace kfr;
  std::list<cv::Mat> imagenes;
  std::list<std::string> audio_paths, img_names;

  // Como todas las imgs son iguales en dims, sacamos datos de cualquiera
  Magick::Image img_info;
  img_info.ping(std::filesystem::directory_iterator(img_dir_path)->path().string());
  size_t rows = img_info.rows(), cols = img_info.columns(), size_vec_pix = rows * cols;

  // Clean de directorios
  std::filesystem::remove_all("hidden_sounds");
  std::filesystem::create_directory("hidden_sounds");

  // Cargamos imagenes en memoria
  for (const auto &it : std::filesystem::directory_iterator(img_dir_path)) {
    const std::string path = it.path().string();
    imagenes.emplace_back(rows, cols,
                          CV_8UC3, cv::Scalar(255, 255, 255));
    imagenes.back() = cv::imread(path, cv::IMREAD_UNCHANGED);

    // Lista de nombres de imagenes, solo datos del CA
    img_names.push_back(path.substr(path.find_first_of('/') + 1,
                                    path.find_last_of('.') - path.find_first_of('/') - 1));
  }

  for (const auto &it : std::filesystem::directory_iterator(audio_dir_path)) {
    audio_paths.push_back(it.path().string());
  }

  // Tenemos que ocultar todas las imagenes
  auto audio_path_it = audio_paths.begin();
  while (!imagenes.empty() && audio_path_it != audio_paths.end()) {
    dft::Audio audio{dft::Audio::MakeFromWAV(*audio_path_it++)};
    size_t cap_audio = audio.traits.maximumNElementsPerTime(),
           fragmentos = audio.fmt.length / audio.traits.preferredSamples(),
           n_ocultar = fragmentos * cap_audio / size_vec_pix;

    // 1 audio guarda todas las imagenes
    if (n_ocultar > imagenes.size())
      n_ocultar = imagenes.size();

    // Guardamos todas las imagenes posibles como un bloque consecutivo
    std::vector<unsigned> pixeles;
    pixeles.reserve(n_ocultar * size_vec_pix);

    auto img_it = imagenes.begin(), img_it_end = imagenes.begin();
    std::advance(img_it_end, n_ocultar);

    std::string out_audio_name{};
    auto img_name_it = img_names.begin(), img_name_it_end = img_names.begin();
    std::advance(img_name_it_end, n_ocultar);
    for (size_t j{}; j < n_ocultar; ++j) {
      std::vector<unsigned> pixeles_curr = mat_to_pixel(*img_it++);
      pixeles.insert(pixeles.end(), pixeles_curr.cbegin(), pixeles_curr.cend());

      // El nombre solo contiene las imgs guardadas
      out_audio_name += *img_name_it++ + '-';
    }
    /* println(pixeles); */

    auto encrypted{encrypt(audio, pixeles)};

    // Eliminamos imagenes ya ocultadas
    imagenes.erase(imagenes.begin(), img_it_end);
    img_names.erase(img_names.begin(), img_name_it_end);

    out_audio_name.pop_back();
    out_audio_name += ".wav";
    /* println(out_audio_name); */

    save_audio(encrypted, "hidden_sounds/" + out_audio_name);
  }

  if (!imagenes.empty())
    throw std::runtime_error("# de audios o tiempo insuficientes");
}

/**
 * Recupera imagenes secretas del CA en un conjunto de audios,
 * Los audios necesitan estar nombrados de forma,
 * dato-dato-...-dato.wav,
 * donde dato = 4.817
 * @param path de directorio de audios
 * @param filas de imagen
 * @param columnas de imagen
 */
void recover(const std::string &audio_dir_path, size_t rows, size_t cols) {
  using namespace kfr;

  size_t size_vec_pix = rows * cols;

  // Cleanup
  std::filesystem::remove_all("recover");
  std::filesystem::create_directory("recover");

  std::list<std::string> audio_paths;
  for (const auto &it : std::filesystem::directory_iterator(audio_dir_path)) {
    audio_paths.push_back(it.path().string());
  }

  auto audio_path_it = audio_paths.begin();
  while (audio_path_it != audio_paths.end()) {
    // Substring que contiene solo data de imagenes CA concatenados
    std::string a = audio_path_it->substr(audio_path_it->find_first_of('/') + 1,
                                          audio_path_it->find_last_of('.') - audio_path_it->find_first_of('/') - 1);

    // Arreglo de nombres de imgs
    std::vector<std::string> data_CA;
    boost::split(data_CA, a, boost::is_any_of("-"));

    size_t n_ocultos = data_CA.size();

    dft::Audio audio{dft::Audio::MakeFromWAV(*audio_path_it++)};
    size_t cap_audio = audio.traits.maximumNElementsPerTime(),
           fragmentos = audio.fmt.length / audio.traits.preferredSamples(),
           n_posibles_ocultos = fragmentos * cap_audio / size_vec_pix;

    /* println(n_ocultos); */

    // # ocultos del path es mayor que lo permitido por audio
    if (n_ocultos > n_posibles_ocultos)
      n_ocultos = n_posibles_ocultos;

    // Recuperamos audios en un bloque consecutivo
    std::vector<unsigned> n_decrypted = decrypt(audio, n_ocultos * size_vec_pix);
    /* println(n_decrypted); */

    // Separacion de bloque en las n imagenes interiores
    auto it_tok = data_CA.begin();
    for (size_t i{}; i < n_decrypted.size(); i += size_vec_pix) {
      cv::Mat img = pixel_to_mat({n_decrypted.begin() + i, n_decrypted.begin() + i + size_vec_pix}, rows, cols);
      /* println(*it_tok); */

      cv::imwrite(std::format("recover/{}.png", *it_tok++), img);
      /* cv::imshow("Desencriptado", img); */
      /* while (cv::waitKey(30) != 'q') { */
      /* } */
    }
  }
}
} // namespace ImgHide
