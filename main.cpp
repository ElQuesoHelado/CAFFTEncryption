#include "CA.hpp"
#include "ImgHide.hpp"

int main() {
  CA::SSC_hide({"pictures128/1.png", "pictures128/2.png", "pictures128/doom1.png", "pictures128/doom2.png"});

  ImgHide::hide("secret", "sounds");

  ImgHide::recover("hidden_sounds", 128, 128);
  CA::SSC_recovery("recover");

  return 0;
}
