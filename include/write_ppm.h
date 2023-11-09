#ifndef WRITE_PPM_H
#define WRITE_PPM_H

#include <vector>
#include <string>

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
// Write an rgb or grayscale image to a .ppm file.
//
// Inputs:
//   filename  path to .ppm file as string
//   data  width*heigh*num_channels array of image intensity data
//   width  image width (i.e., number of columns)
//   height  image height (i.e., number of rows)
//   num_channels  number of channels (e.g., for rgb 3, for grayscale 1)
// Returns true on success, false on failure (e.g., can't open file)
bool write_ppm(
  const std::string & filename,
  const std::vector<unsigned char> & data,
  const int width,
  const int height,
  const int num_channels){
    assert((num_channels == 3 || num_channels == 1) &&
           ".ppm only supports RGB or grayscale images");
    int rgb[3];
    int c = 0;
    if (num_channels == 3) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "P3\n";
            file << width << " " << height << "\n";
            file << "255\n";

            for (const auto &val : data) {
                if (c == 3) {
                    file << rgb[0] << " " << rgb[1] << " " << rgb[2] << " \n";
                    c = 0;
                }
                rgb[c] = (int)val;
                c++;
            }
            // write out the last pixel in the buffer
            file << rgb[0] << " " << rgb[1] << " " << rgb[2] << " \n";
            file.close();
            return true;
        } else
            return false;
    } else { // num_channels == 1
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "P2\n";
            file << width << " " << height << "\n";
            file << "255\n";

            for (const auto &val : data)
                file << (int)val << " \n";
            file.close();
            return true;
        } else
            return false;
    }
    return false;
  }

#endif
