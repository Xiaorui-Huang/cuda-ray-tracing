#include "write_ppm.h"

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
using namespace std;

bool write_ppm(const std::string &filename, const std::vector<unsigned char> &data, const int width, const int height,
               const int num_channels) {
    assert((num_channels == 3 || num_channels == 1) && ".ppm only supports RGB or grayscale images");
    array<int, 3> rgb;
    int c = 0;
    if (num_channels == 3) {
        ofstream file(filename);
        if (file.is_open()) {
            file << "P3\n";
            file << width << " " << height << "\n";
            file << "255\n";

            for (const auto &val : data) {
                if (c == 3) {
                    file << rgb.at(0) << " " << rgb.at(1) << " " << rgb.at(2) << " \n";
                    c = 0;
                }
                rgb.at(c) = (int)val;
                c++;
            }
            // write out the last pixel in the buffer
            file << rgb.at(0) << " " << rgb.at(1) << " " << rgb.at(2) << " \n";
            file.close();
            return true;
        } else
            return false;
    } else { // num_channels == 1
        ofstream file(filename);
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
