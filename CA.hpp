#include <algorithm>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <iostream>
/* #include <opencv2/core/hal/interface.h> */
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/core/matx.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <ostream>
#include <random>
// #include <iterator>
// #include <ImageMagick-7/Magick++.h>
#include <Magick++.h>
// #include <ImageMagick-7/magick/
// #include "ImageMagick-7/Magick++/Image.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

// #include <boost/gil/extension/io/jpeg.hpp>

#define modulo(x, m) (((x % m) + m) % m)

// Detalles de implementacion
namespace {
/*
 * Encuentra la representacion numerica de un triplete RGB
 */
int vec3b_to_int(const cv::Vec3b &pixel) {
  int res{pixel[2]};
  res = res << 8;
  res += pixel[1];
  res = res << 8;
  res += pixel[0];
  return res;
}

cv::Vec3b int_to_vec3b(int num) {
  cv::Vec3b vec;
  vec[0] = num;
  num = num >> 8;
  vec[1] = num;
  num = num >> 8;
  vec[2] = num;
  return vec;
}

/*
 * Funcion de transicion para un estado del espacio
 * Se aplica a todos los vecinos, de acuerdo a un numero w(0-511)
 */
int trans_celda(unsigned short w, const size_t &i, const size_t &j, const cv::Mat &space) {
  int res{};
  for (int k = 1; k >= -1; --k) {
    for (int l = 1; l >= -1; --l) {
      res +=
          (w & 1) * space.at<uchar>(modulo(i + k, space.rows),
                                    modulo(j + l, space.cols));
      w = w >> 1;
    }
  }
  return res;
}

/*
 * Funcion de transicion para un estado del espacio
 * Se aplica a todos los vecinos, de acuerdo a un numero w(0-511)
 */
int trans_celda_vec3b(unsigned short w, const size_t &i, const size_t &j, const cv::Mat &space) {
  int res{};
  for (int k = 1; k >= -1; --k) {
    for (int l = 1; l >= -1; --l) {
      res +=
          (w & 1) *
          vec3b_to_int(space.at<cv::Vec3b>(modulo(i + k, space.rows),
                                           modulo(j + l, space.cols)));
      w = w >> 1;
    }
  }
  return res;
}

void gen_rand_img(cv::Mat &imagen) {
  // Se crea generador de numeros random
  // Numero "verdaderamente" random para seed se pseudo gen
  std::random_device trueRand;
  std::mt19937 eng(trueRand());
  std::uniform_int_distribution<size_t> uniformDist(0, 255);

  // Generador usando std::bind, sintax mas corta
  std::function<size_t(void)> generator = std::bind(uniformDist, eng);

  unsigned char *ptr = imagen.data;
  for (size_t cell = 1; cell < imagen.rows * imagen.cols; ++cell, ++ptr) {
    *ptr = generator();
  }
}

} // namespace

namespace CA {
/*
 * Funcion que realiza una Linear Celular Automata en una imagen de 32x32 pixeles
 * @param tiempos El numero de tiempos que va a evolucionar
 */
void LCA(size_t tiempos) {
  // Inicializamos las imagenes para cada evolucion
  std::vector<cv::Mat> espacios;
  unsigned short regla;

  espacios.reserve(tiempos + 1);

  for (size_t i{}; i < tiempos + 1; ++i) { // Evita que openCV haga referencia a una sola Mat
    espacios.emplace_back(32, 32,
                          CV_8UC1, cv::Scalar(255, 255, 255));
  }

  // Generacion de numero regla para cada espacio
  std::random_device trueRand;
  std::mt19937 eng(trueRand());
  std::uniform_int_distribution<unsigned short> dist(0, 511);

  auto generador = [&eng, &dist]() { return dist(eng); };
  // auto generador = [&eng, &dist]() { return 511; };

  regla = generador();
  // std::generate(reglas.begin(), reglas.end(), generador);
  // Se carga tiempo 0 con una configuracion predefinida
  espacios[0] = cv::imread("pictures/conf1.png", cv::IMREAD_GRAYSCALE);

  // Realizamos evoluciones
  for (size_t i{}; i < tiempos; ++i) {
    // Se aplica funcion de transicion a todas las celdas
    for (size_t row{}; row < espacios[i].rows; ++row) {
      for (size_t col{}; col < espacios[i].cols; ++col) {
        espacios[i + 1].at<unsigned char>(row, col) =
            (trans_celda(regla, row, col, espacios[i]) % 2) ? 255 : 0;
      }
    }
  }

  cv::Mat out;
  cv::hconcat(espacios, out);
  cv::imshow("Evoluciones", out);
  cv::waitKey(0);
}

/*
 * Linear Celular Automata con memoria(orden 3) invertible en una imagen de 32x32 pixeles
 * @param tiempos El numero de tiempos que va a evolucionar
 */
void LMCA(size_t tiempos) {
  // Inicializamos las imagenes para cada evolucion
  std::vector<cv::Mat> espacios;
  std::vector<unsigned short> reglas(2); // Para ir de t -> t+1 solo necesitamos 2 reglas

  espacios.reserve(tiempos + 4);

  for (size_t i{}; i < tiempos + 4; ++i) { // Evita que openCV haga referencia a una sola Mat
    espacios.emplace_back(32, 32,
                          CV_8UC1, cv::Scalar(255, 255, 255));
  }

  // Generacion de numero regla para cada espacio
  std::random_device trueRand;
  std::mt19937 eng(trueRand());
  std::uniform_int_distribution<unsigned short> dist(0, 511);

  auto generador = [&eng, &dist]() { return dist(eng); };
  // auto generador = [&eng, &dist]() { return 511; };

  std::generate(reglas.begin(), reglas.end(), generador);

  // Valores iniciales 3 configuraciones
  espacios[0] = cv::imread("pictures/conf1.png", cv::IMREAD_GRAYSCALE);
  espacios[1] = cv::imread("pictures/conf2.png", cv::IMREAD_GRAYSCALE);
  espacios[2] = cv::imread("pictures/conf3.png", cv::IMREAD_GRAYSCALE);

  // Realizamos evoluciones
  for (size_t i{3}; i < tiempos + 4; ++i) {
    // Se aplica funcion de transicion a todas las celdas
    for (size_t row{}; row < espacios[i].rows; ++row) {
      for (size_t col{}; col < espacios[i].cols; ++col) {
        espacios[i].at<unsigned char>(row, col) =
            ((trans_celda(reglas[0], row, col, espacios[i - 1]) +
              trans_celda(reglas[1], row, col, espacios[i - 2]) +
              espacios[i - 3].at<unsigned char>(row, col)) %
             2)
                ? 255
                : 0;
      }
    }
  }

  cv::Mat out;
  cv::hconcat(espacios, out);
  cv::imshow("Evoluciones", out);
  cv::waitKey(0);
}

/*
 * LMCA especificada para compartir imagenes secretas
 * @param img_paths Direccion de las imagenes para ser compartidas
 */
void SSC_hide(std::initializer_list<std::string> img_paths) {
  // Se encuentra las dimensiones maximas y el mayor bit depth
  // Magick::Image.ping(...) evita cargar las imagenes en memoria
  size_t maxRows{}, maxCols{}, depth{}, n = img_paths.size();
  for (Magick::Image img_info; const auto &img_name : img_paths) {
    img_info.ping(img_name);
    if (img_info.rows() > maxRows)
      maxRows = img_info.rows();
    if (img_info.columns() > maxCols)
      maxCols = img_info.columns();
    // std::cout << std::format("Name: {} Rows: {}, Cols: {} Depth:{} \n",
    //                          img_name,
    //                          img_info.rows(), img_info.columns(), img_info.channels());
    // std::cout << img_info. << std::endl;
  }

  // Numero random l, hacemos n + l - 1  evoluciones
  size_t l = 10; //*****Cambiar a random

  // Inicializamos las imagenes para cada evolucion
  std::vector<cv::Mat> espacios;
  std::vector<unsigned short> reglas(n);

  espacios.reserve(n + l);

  // Creamos imagenes blancas de dimensiones maximas, sirven de padding
  for (size_t i{}; i <= n + l - 1; ++i) { //***CV_8UC3, que pasa si todas son blanco y negro???
    espacios.emplace_back(maxRows, maxCols,
                          CV_8UC3, cv::Scalar(255, 255, 255));
  }

  // Insertamos las n imagenes en los n primeros espacios, de acuerdo al padding
  {
    size_t i = 1;
    Magick::Image img_info;
    for (cv::Mat curr_mat; const auto img_name : img_paths) {
      img_info.ping(img_name);
      cv::imread(img_name).copyTo(
          espacios[i](cv::Rect((maxCols - img_info.columns()) / 2,
                               (maxRows - img_info.rows()) / 2,
                               img_info.columns(), img_info.rows())));
      ++i;
    }
  }

  // Generacion de numero regla para cada espacio
  std::random_device trueRand;
  std::mt19937 eng(trueRand());
  std::uniform_int_distribution<unsigned short> dist(0, 511);

  auto generador = [&eng, &dist]() { return dist(eng); };
  // // auto generador = [&eng, &dist]() { return 511; };

  std::generate(reglas.begin(), reglas.end(), generador);

  // Generamos nuestra configuracion 0, es una imagen con valores random
  cv::randu(espacios[0], 0, 256);

  // Se determina la cantidad de color representables con el bit depth mas grande
  //****** cambiar a determinar bits
  unsigned int n_colores = std::pow(2, 24);

  // Realizamos evoluciones usando un orden n+1
  for (size_t i{n + 1}; i <= n + l - 1; ++i) {
    // Se aplica funcion de transicion a todas las celdas
    for (size_t row{}; row < maxRows; ++row) {
      for (size_t col{}; col < maxCols; ++col) {

        int sum{};
        // Aplicamos la funcion de transicion en orden n+1, excepto el ultimo
        for (size_t curr = i - 1, r = 0; curr >= i - n; --curr, ++r) {
          sum += trans_celda_vec3b(reglas[r], row, col, espacios[curr]);
          sum %= n_colores;
        }

        // Solo sumamos el estado del ultimo espacio
        sum += vec3b_to_int(espacios[i - n - 1].at<cv::Vec3b>(row, col));
        sum %= n_colores;

        // Convertimos la representacion numerica del pixel a vec3b en el siguiente tiempo
        espacios[i].at<cv::Vec3b>(row, col) = int_to_vec3b(sum);
      }
    }
  }

  // Eliminamos sombras creadas anteriormente
  std::filesystem::remove_all("secret");
  // std::filesystem::remove_all("public");

  // std::filesystem::create_directory("shadows");
  std::filesystem::create_directory("secret");

  // Guardamos las ultimas n sombras para poder recuperar imagenes originales
  // #participante.w.png
  for (size_t i{l}, m{1}; i < espacios.size(); ++i, ++m) {
    cv::imwrite(std::format("secret/{}.{}.png", m, reglas[m - 1]), espacios[i]);
  }

  // Publicamos ultima configuracion inicial y numero l
  cv::imwrite(std::format("secret/0.{}.png", l), espacios[l - 1]);

  // Mostrar en pantalla, en un grid
  size_t out_dim = std::ceil(std::sqrt(espacios.size()));
  cv::Mat grid(maxRows * out_dim, maxCols * out_dim, CV_8UC3, cv::Scalar(255, 255, 255));

  size_t j{};
  for (size_t i{}; i < out_dim; ++i) {
    for (size_t k{}; k < out_dim && j < espacios.size(); ++k) {
      cv::Rect roi(k * maxCols, i * maxRows, maxCols, maxRows);
      espacios[i * out_dim + k].copyTo(grid(roi));
      ++j;
    }
  }

  cv::namedWindow("Evoluciones", cv::WINDOW_NORMAL);
  cv::imshow("Evoluciones", grid);
  cv::resizeWindow("Evoluciones", 1000, 1000);
  while (cv::waitKey(30) != 'q') {
  }
}

/*
 * Funcion que recupera las configuraciones originales
 * @param directorio que contiene los secretos
 */
void SSC_recovery(std::string secret_dir_path) {
  size_t n,
      rows, cols, l;
  std::vector<cv::Mat> espacios;
  std::vector<unsigned short> reglas;
  std::set<std::string> names;

  std::regex delim(".");
  /*
   * Leemos todos los archivos dentro de secret_dir_path y los ordenamos
   */
  for (auto x : std::filesystem::directory_iterator(secret_dir_path)) {
    names.insert(x.path().string());
    // //*********Agregar excepciones

    // // Magick::Image img_info;
    // names[0].star

    // std::cout << x.path().string() << std::endl;
    // ++n;
  }

  //*****************
  // Conf 0 se encuentra en el principio del set
  // Inicializamos variables en base a esta
  n = names.size() - 1;
  { // Numero l
    std::string non_prefix = names.begin()->substr(secret_dir_path.size() + 1);

    size_t punt1 = non_prefix.find('.');
    size_t punt2 = non_prefix.rfind('.');

    std::string numero = non_prefix.substr(punt1 + 1, punt2 - 2);
    // "l"
    l = std::stoi(numero);
  }

  // std::cout << l << std::endl;
  reglas.resize(n, 0);
  espacios.reserve(n + l);

  // Cargamos configuraciones y numeros regla
  {
    auto it = names.crbegin();
    for (size_t i = 0; i < n; ++i) {
      espacios.emplace_back(cv::imread(*it,
                                       cv::IMREAD_UNCHANGED));

      std::string non_prefix = it->substr(secret_dir_path.size() + 1);

      size_t punt1 = non_prefix.find('.');
      size_t punt2 = non_prefix.rfind('.');

      std::string numero = non_prefix.substr(punt1 + 1, punt2 - 2);

      reglas[i] = std::stoi(numero);
      it++;
    }
  }

  espacios.emplace_back(cv::imread(*names.begin(),
                                   cv::IMREAD_UNCHANGED));

  // Llenamos los siguientes espacios con imagenes en blanco
  rows = espacios[0].rows;
  cols = espacios[0].cols;
  for (size_t i = n + 1; i < n + l; ++i) {
    espacios.emplace_back(rows, cols,
                          CV_8UC3, cv::Scalar(255, 255, 255));
  }

  // Hallar bit depth
  //****** cambiar a usar color de opencv
  unsigned int n_colores = std::pow(2, 24);

  // Realizamos evoluciones usando un orden n+1
  for (size_t i{n + 1}; i <= n + l - 1; ++i) {
    // Se aplica funcion de transicion a todas las celdas
    for (size_t row{}; row < rows; ++row) {
      for (size_t col{}; col < cols; ++col) {

        int sum{};
        // Aplicamos la funcion de transicion inversa en orden n+1, excepto el ultimo
        for (size_t curr = i - 1, r = 0; curr >= i - n; --curr, ++r) {
          sum -= trans_celda_vec3b(reglas[r], row, col, espacios[curr]);
          sum = modulo(sum, n_colores);
        }

        // Solo sumamos el estado del ultimo espacio
        sum += vec3b_to_int(espacios[i - n - 1].at<cv::Vec3b>(row, col));
        sum = modulo(sum, n_colores);

        // Convertimos la representacion numerica del pixel a vec3b en el siguiente tiempo
        espacios[i].at<cv::Vec3b>(row, col) = int_to_vec3b(sum);
      }
    }
  }

  std::filesystem::remove_all("pictures_recover");

  std::filesystem::create_directory("pictures_recover");

  for (size_t i{l - 1}; i < espacios.size() - 1; ++i) {
    cv::imwrite(std::format("pictures_recover/{}.png", i), espacios[i]);
  }

  // Grid
  size_t out_dim = std::ceil(std::sqrt(espacios.size()));
  cv::Mat grid(rows * out_dim, cols * out_dim, CV_8UC3, cv::Scalar(255, 255, 255));

  size_t j{};
  for (size_t i{}; i < out_dim; ++i) {
    for (size_t k{}; k < out_dim && j < espacios.size(); ++k) {
      cv::Rect roi(k * cols, i * rows, cols, rows);
      espacios[i * out_dim + k].copyTo(grid(roi));
      ++j;
    }
  }
  cv::namedWindow("Evoluciones", cv::WINDOW_NORMAL);
  cv::imshow("Evoluciones", grid);
  cv::resizeWindow("Evoluciones", 1000, 1000);
  while (cv::waitKey(30) != 'q') {
  }
}
} // namespace CA
